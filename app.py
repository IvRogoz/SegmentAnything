import os
import uuid
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from ultralytics import SAM
from PIL import Image
from simple_lama_inpainting import SimpleLama
from zipdepth.inference.predictor import DepthInference
import base64
import io
import warnings
warnings.filterwarnings("ignore")

os.makedirs("uploads", exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


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


def segment_image(image_path):
    results = model(image_path, device=device, verbose=False, max_det=50)
    masks = []
    scores = []
    if results[0].masks is not None:
        for i, mask in enumerate(results[0].masks.data):
            masks.append(mask.cpu().numpy())
            score = float(results[0].boxes.conf[i]) if results[0].boxes is not None and len(results[0].boxes) > i else 0.0
            scores.append(score)
    masks, scores = remove_duplicate_masks(masks, scores)
    return masks, scores, results[0].orig_shape


def estimate_depth(image_path):
    """Return ZipDepth's continuous relative depth map, normalized for the viewer."""
    depth = depth_model.infer_image(cv2.imread(image_path))
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

@app.route("/")
def index():
    return render_template("index.html")

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
    if mode not in {"segmentation", "depth"}:
        return jsonify({"error": "Invalid mode"}), 400
    input_size = int(request.form.get("input_size", 384))
    if input_size not in {384, 512, 768}:
        return jsonify({"error": "Invalid ZipDepth input size"}), 400

    try:
        if mode == "depth":
            depth_model.input_size = input_size
            depth_map, shape = estimate_depth(filepath)
            masks, scores = [], []
        else:
            masks, scores, shape = segment_image(filepath)
        orig = cv2.imread(filepath)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        h, w = shape[:2]

        mask_list = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_uint8 = (mask * 255).astype(np.uint8)
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

        if mode == "segmentation":
            mask_list.sort(key=lambda m: m["area"], reverse=True)

        if mode == "segmentation":
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks:
                combined_mask = np.maximum(combined_mask, (m > 0).astype(np.uint8) * 255)

            k = max(3, int(min(h, w) * 0.015) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_dilated = cv2.dilate(combined_mask, kernel, iterations=1)
            mask_dilated = cv2.GaussianBlur(mask_dilated.astype(np.float32), (0, 0), k * 0.6)
            mask_dilated = np.clip(mask_dilated, 0, 255).astype(np.uint8)
            inpainted_rgb = np.asarray(
                inpaint_model(Image.fromarray(orig_rgb), Image.fromarray(mask_dilated))
            ).copy()[:h, :w]
        else:
            inpainted_rgb = orig_rgb
        inpainted_enc = encode_pil(Image.fromarray(inpainted_rgb))

        orig_enc = encode_pil(Image.fromarray(orig_rgb))
        depth_enc = None
        if mode == "depth":
            depth_preview = Image.fromarray((depth_map * 255).astype(np.uint8))
            depth_enc = f"data:image/png;base64,{encode_pil(depth_preview)}"

        return jsonify({
            "success": True,
            "mode": mode,
            "image_id": filename,
            "image": f"data:image/png;base64,{orig_enc}",
            "background": f"data:image/png;base64,{inpainted_enc}",
            "masks": mask_list,
            "depth": depth_enc,
            "shape": [h, w]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "0")))
