import os
import uuid
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from ultralytics import SAM
from PIL import Image
from simple_lama_inpainting import SimpleLama
from zipdepth.inference.predictor import DepthInference
import base64
import io
import json
import threading
import time
import warnings
warnings.filterwarnings("ignore")

os.makedirs("uploads", exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
video_sources = {}
edgetam_video_sessions = {}
edgetam_model_lock = threading.Lock()
edgetam_inference_lock = threading.Lock()
edgetam_image_predictor = None
edgetam_video_predictor = None
progress_jobs = {}
progress_jobs_lock = threading.Lock()


def set_job_progress(job_id, percent, label="Processing"):
    if not job_id:
        return
    now = time.monotonic()
    with progress_jobs_lock:
        expired = [
            key for key, value in progress_jobs.items()
            if now - value["updated"] > 300
        ]
        for key in expired:
            progress_jobs.pop(key, None)
        progress_jobs[job_id] = {
            "percent": max(0, min(100, int(round(percent)))),
            "label": label,
            "updated": now,
        }


@app.route("/progress/<job_id>")
def job_progress(job_id):
    with progress_jobs_lock:
        status = progress_jobs.get(job_id)
    if status is None:
        return jsonify({"percent": 0, "label": "Starting EdgeTAM"})
    return jsonify({"percent": status["percent"], "label": status["label"]})


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify({"error": "Upload exceeds the server limit. Choose Local Video to use the original file without uploading it."}), 413


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SAM("sam2.1_s.pt")
inpaint_model = SimpleLama(torch.device(device))
depth_model = DepthInference(
    checkpoint_path=os.path.join("vendor", "ZipDepth", "checkpoints", "zipdepth_base.pth"),
    device=device,
    input_size=384,
    warmup_iters=1,
)


class LazyOpenCVVideoFrames:
    """Decode only the EdgeTAM frame currently requested; never extract frames to disk."""

    def __init__(self, filepath, image_size, offload_video_to_cpu, compute_device):
        self.filepath = filepath
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.compute_device = compute_device
        self.capture = cv2.VideoCapture(filepath)
        if not self.capture.isOpened():
            raise RuntimeError("EdgeTAM cannot open the video")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.next_frame = 0
        self.lock = threading.Lock()
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    def __len__(self):
        return self.frame_count

    def __getitem__(self, index):
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        with self.lock:
            if index != self.next_frame:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.capture.read()
            self.next_frame = index + 1
        if not ok:
            raise RuntimeError(f"EdgeTAM cannot decode video frame {index}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        if not self.offload_video_to_cpu:
            tensor = tensor.to(self.compute_device, non_blocking=True)
        return tensor

    def close(self):
        with self.lock:
            self.capture.release()


def load_edgetam_video_frames(video_path, image_size, offload_video_to_cpu, **kwargs):
    loader = LazyOpenCVVideoFrames(
        video_path,
        image_size,
        offload_video_to_cpu,
        kwargs.get("compute_device", torch.device("cuda")),
    )
    return loader, loader.video_height, loader.video_width


def get_edgetam_models():
    global edgetam_image_predictor, edgetam_video_predictor
    if not torch.cuda.is_available():
        raise RuntimeError("EdgeTAM requires CUDA; CPU fallback is disabled")
    if edgetam_image_predictor is not None and edgetam_video_predictor is not None:
        return edgetam_image_predictor, edgetam_video_predictor
    with edgetam_model_lock:
        if edgetam_image_predictor is None or edgetam_video_predictor is None:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import sam2.sam2_video_predictor as video_predictor_module

            edge_root = os.path.join("vendor", "EdgeTAM")
            checkpoint = os.path.join(edge_root, "checkpoints", "edgetam.pt")
            video_predictor_module.load_video_frames = load_edgetam_video_frames
            image_model = build_sam2(
                "edgetam.yaml",
                checkpoint,
                device="cuda",
                apply_postprocessing=False,
            )
            edgetam_image_predictor = SAM2ImagePredictor(image_model)
            edgetam_video_predictor = build_sam2_video_predictor(
                "edgetam.yaml",
                checkpoint,
                device="cuda",
            )
    return edgetam_image_predictor, edgetam_video_predictor


def segment_image_edgetam(image, box, job_id=None):
    image_predictor, _ = get_edgetam_models()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with edgetam_inference_lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        set_job_progress(job_id, 20, "Encoding EdgeTAM frame")
        image_predictor.set_image(rgb)
        set_job_progress(job_id, 70, "Segmenting selected object")
        masks, scores, _logits = image_predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        selected_mask = masks[best].astype(np.float32)
        selected_score = float(scores[best])
        image_predictor.reset_predictor()
    set_job_progress(job_id, 95, "Encoding selected object")
    return [selected_mask], [selected_score], image.shape[:2]


def release_edgetam_video_session(video_id):
    session = edgetam_video_sessions.pop(video_id, None)
    if session is None:
        return
    images = session["state"].get("images")
    if hasattr(images, "close"):
        images.close()
    del session
    torch.cuda.empty_cache()


def clear_edgetam_video_sessions():
    for video_id in list(edgetam_video_sessions):
        release_edgetam_video_session(video_id)


def segment_video_frame_edgetam(video_id, frame_index, box=None, job_id=None):
    filepath = video_sources[video_id]
    with edgetam_inference_lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        session = edgetam_video_sessions.get(video_id)
        if session is None:
            if box is None:
                raise ValueError("EdgeTAM requires a bounding box on frame 1")
            _image_predictor, predictor = get_edgetam_models()
            set_job_progress(job_id, 20, "Initializing EdgeTAM video")
            state = predictor.init_state(
                video_path=filepath,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            set_job_progress(job_id, 60, "Selecting object from bounding box")
            _output_frame, _object_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=0,
                box=np.asarray(box, dtype=np.float32),
            )
            initial_masks = [(mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.float32)]
            scores = [1.0]
            set_job_progress(job_id, 95, "Object selected for tracking")
            session = {
                "predictor": predictor,
                "state": state,
                "initial_masks": initial_masks,
                "scores": scores,
                "last_frame": 0,
            }
            edgetam_video_sessions[video_id] = session

        if frame_index == 0:
            return session["initial_masks"], session["scores"]
        if not session["initial_masks"]:
            return [], []

        predictor = session["predictor"]
        state = session["state"]
        if frame_index > session["last_frame"]:
            start_frame = session["last_frame"] + 1
            frames_to_track = frame_index - start_frame
        else:
            start_frame = frame_index
            frames_to_track = 0

        target_masks = None
        expected_outputs = max(1, frames_to_track + 1)
        completed_outputs = 0
        for output_frame, _object_ids, mask_logits in predictor.propagate_in_video(
            state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=frames_to_track,
        ):
            completed_outputs += 1
            set_job_progress(
                job_id,
                99 * completed_outputs / expected_outputs,
                "Propagating EdgeTAM masks",
            )
            if output_frame == frame_index:
                target_masks = [
                    (mask_logits[index, 0] > 0.0).cpu().numpy().astype(np.float32)
                    for index in range(mask_logits.shape[0])
                ]
        if target_masks is None:
            raise RuntimeError(f"EdgeTAM did not produce frame {frame_index}")
        session["last_frame"] = max(session["last_frame"], frame_index)
        return target_masks, session["scores"]


def remove_duplicate_masks(masks, scores):
    """Keep one mask when SAM returns effectively same object multiple times."""
    kept_masks = []
    kept_scores = []
    for index in np.argsort(scores)[::-1]:
        candidate = masks[index] > 0.5
        candidate_area = int(candidate.sum())
        if candidate_area == 0:
            continue

        duplicate = False
        for existing in kept_masks:
            intersection = int(np.logical_and(candidate, existing > 0.5).sum())
            if intersection == 0:
                continue
            union = candidate_area + int((existing > 0.5).sum()) - intersection
            iou = intersection / union
            containment = intersection / min(candidate_area, int((existing > 0.5).sum()))
            if iou >= 0.85 or containment >= 0.95:
                duplicate = True
                break

        if not duplicate:
            kept_masks.append(masks[index])
            kept_scores.append(scores[index])
    return kept_masks, kept_scores


def segment_image(image):
    results = model(image, device=device, verbose=False, max_det=50)
    masks = []
    scores = []
    if results[0].masks is not None:
        for i, mask in enumerate(results[0].masks.data):
            masks.append(mask.cpu().numpy())
            score = float(results[0].boxes.conf[i]) if results[0].boxes is not None and len(results[0].boxes) > i else 0.0
            scores.append(score)
    masks, scores = remove_duplicate_masks(masks, scores)
    return masks, scores, results[0].orig_shape


def estimate_depth(image):
    """Return ZipDepth's continuous relative depth map, normalized for the viewer."""
    depth = depth_model.infer_image(image)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    normalized = (depth - depth_min) / max(depth_max - depth_min, 1e-6)
    return normalized, depth.shape

def encode_pil(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def rgba_overlay(orig_rgb, mask, alpha=0.4):
    mask_bool = mask > 0
    overlay = orig_rgb.copy()
    color = np.random.randint(50, 255, 3, dtype=np.uint8)
    for c in range(3):
        overlay[:, :, c] = np.where(mask_bool, (overlay[:, :, c] * (1 - alpha) + color[c] * alpha).astype(np.uint8), overlay[:, :, c])
    return overlay

def extract_object_rgba(orig_rgb, mask, feather_radius=2):
    h, w = orig_rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    if feather_radius > 0 and np.any(mask_uint8 > 0):
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (0, 0), feather_radius)
    for c in range(3):
        rgba[:, :, c] = orig_rgb[:, :, c]
    rgba[:, :, 3] = (mask_uint8 * (mask > 0.5)).astype(np.uint8)
    return rgba


def infer_image_array(orig, mode, input_size, image_id, segmentation_result=None, inpainting=False, job_id=None, edgetam_box=None):
    """Infer from an already decoded OpenCV BGR image; never writes a frame to disk."""
    if mode == "depth":
        depth_model.input_size = input_size
        depth_map, shape = estimate_depth(orig)
        masks, scores = [], []
    elif segmentation_result is not None:
        masks, scores = segmentation_result
        shape = orig.shape[:2]
    elif mode == "edgetam":
        if edgetam_box is None:
            raise ValueError("EdgeTAM requires a bounding box")
        masks, scores, shape = segment_image_edgetam(orig, edgetam_box, job_id=job_id)
    else:
        masks, scores, shape = segment_image(orig)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = shape[:2]

    mask_list = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        overlay_img = Image.fromarray(rgba_overlay(orig_rgb, mask))
        obj_rgba = extract_object_rgba(orig_rgb, mask)
        area = int(np.sum(mask > 0))
        mask_list.append({
            "id": i,
            "overlay": f"data:image/png;base64,{encode_pil(overlay_img)}",
            "object": f"data:image/png;base64,{encode_pil(Image.fromarray(obj_rgba, mode='RGBA'))}",
            "area": area,
            "score": round(score, 3)
        })
    if mode != "depth" and inpainting:
        mask_list.sort(key=lambda m: m["area"], reverse=True)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8) * 255)
        k = max(3, int(min(h, w) * 0.015) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_dilated = cv2.dilate(combined_mask, kernel, iterations=1)
        mask_dilated = cv2.GaussianBlur(mask_dilated.astype(np.float32), (0, 0), k * 0.6)
        mask_dilated = np.clip(mask_dilated, 0, 255).astype(np.uint8)
        inpainted_rgb = np.asarray(inpaint_model(Image.fromarray(orig_rgb), Image.fromarray(mask_dilated))).copy()[:h, :w]
    else:
        inpainted_rgb = orig_rgb

    depth_enc = None
    if mode == "depth":
        depth_preview = Image.fromarray((depth_map * 255).astype(np.uint8))
        depth_enc = f"data:image/png;base64,{encode_pil(depth_preview)}"
    return {
        "success": True,
        "mode": mode,
        "image_id": image_id,
        "image": f"data:image/png;base64,{encode_pil(Image.fromarray(orig_rgb))}",
        "background": f"data:image/png;base64,{encode_pil(Image.fromarray(inpainted_rgb))}",
        "masks": mask_list,
        "depth": depth_enc,
        "shape": [h, w]
    }

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file"}), 400

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    video_id = str(uuid.uuid4())
    filepath = os.path.join("uploads", f"{video_id}{ext}")
    file.save(filepath)
    return register_video_source(filepath)


def register_video_source(filepath):
    clear_edgetam_video_sessions()
    video_id = str(uuid.uuid4())
    capture = cv2.VideoCapture(filepath)
    if not capture.isOpened():
        capture.release()
        return jsonify({"error": "Unsupported video"}), 400
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frame_count < 1 or fps <= 0 or width < 1 or height < 1:
        return jsonify({"error": "Video has no readable frames"}), 400
    video_sources[video_id] = filepath
    return jsonify({
        "video_id": video_id,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    })


@app.route("/video/select-local", methods=["POST"])
def select_local_video():
    """Open a native Windows picker and keep the original video path only."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.askopenfilename(
            title="Choose Local Video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception as error:
        return jsonify({"error": f"Cannot open the native video picker: {error}"}), 500

    if not filepath:
        return jsonify({"error": "No video selected"}), 400
    return register_video_source(filepath)


def read_video_frame(video_id, frame_index):
    filepath = video_sources.get(video_id)
    if not filepath or not os.path.isfile(filepath):
        return None, (jsonify({"error": "Video is unavailable"}), 404)
    capture = cv2.VideoCapture(filepath)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= frame_count:
        capture.release()
        return None, (jsonify({"error": "Frame is outside video"}), 400)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None, (jsonify({"error": "Cannot decode video frame"}), 500)
    return frame, None


def parse_edgetam_box(raw_box, width, height):
    if not raw_box:
        return None
    try:
        values = [float(value) for value in json.loads(raw_box)]
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("Invalid EdgeTAM bounding box")
    if len(values) != 4 or not all(np.isfinite(values)):
        raise ValueError("Invalid EdgeTAM bounding box")
    x1, y1, x2, y2 = values
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("EdgeTAM bounding box must have positive width and height")
    return [x1, y1, x2, y2]


@app.route("/video/<video_id>/frame/<int:frame_index>")
def video_frame(video_id, frame_index):
    frame, error = read_video_frame(video_id, frame_index)
    if error:
        return error
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return jsonify({"error": "Cannot encode video frame"}), 500
    return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")


@app.route("/video/<video_id>/infer/<int:frame_index>", methods=["POST"])
def infer_video_frame(video_id, frame_index):
    frame, error = read_video_frame(video_id, frame_index)
    if error:
        return error
    mode = request.form.get("mode", "segmentation")
    if mode not in {"segmentation", "depth", "edgetam"}:
        return jsonify({"error": "Invalid mode"}), 400
    input_size = int(request.form.get("input_size", 384))
    inpainting = request.form.get("inpainting", "false").lower() == "true"
    job_id = request.form.get("job_id")
    set_job_progress(job_id, 0, "Starting EdgeTAM")
    if input_size not in {384, 512, 768}:
        return jsonify({"error": "Invalid ZipDepth input size"}), 400
    try:
        if mode == "edgetam":
            box = parse_edgetam_box(request.form.get("box"), frame.shape[1], frame.shape[0])
            masks, scores = segment_video_frame_edgetam(
                video_id,
                frame_index,
                box=box,
                job_id=job_id,
            )
            result = infer_image_array(
                frame,
                mode,
                input_size,
                f"{video_id}:{frame_index}",
                segmentation_result=(masks, scores),
                inpainting=inpainting,
                job_id=job_id,
            )
            set_job_progress(job_id, 100, "EdgeTAM complete")
            return jsonify(result)
        release_edgetam_video_session(video_id)
        return jsonify(infer_image_array(
            frame,
            mode,
            input_size,
            f"{video_id}:{frame_index}",
            inpainting=inpainting,
        ))
    except Exception as e:
        set_job_progress(job_id, 100, "EdgeTAM failed")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file"}), 400

    ext = os.path.splitext(file.filename)[1] or ".jpg"
    filename = str(uuid.uuid4()) + ext
    filepath = os.path.join("uploads", filename)
    file.save(filepath)
    mode = request.form.get("mode", "segmentation")
    if mode not in {"segmentation", "depth", "edgetam"}:
        return jsonify({"error": "Invalid mode"}), 400
    input_size = int(request.form.get("input_size", 384))
    inpainting = request.form.get("inpainting", "false").lower() == "true"
    job_id = request.form.get("job_id")
    set_job_progress(job_id, 0, "Starting EdgeTAM")
    if input_size not in {384, 512, 768}:
        return jsonify({"error": "Invalid ZipDepth input size"}), 400

    try:
        orig = cv2.imread(filepath)
        if orig is None:
            return jsonify({"error": "Cannot decode image"}), 400
        box = parse_edgetam_box(request.form.get("box"), orig.shape[1], orig.shape[0])
        if mode == "edgetam" and box is None:
            return jsonify({"error": "EdgeTAM requires a bounding box"}), 400
        result = infer_image_array(
            orig,
            mode,
            input_size,
            filename,
            inpainting=inpainting,
            job_id=job_id,
            edgetam_box=box,
        )
        set_job_progress(job_id, 100, "EdgeTAM complete")
        return jsonify(result)

    except Exception as e:
        set_job_progress(job_id, 100, "EdgeTAM failed")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "0")))
