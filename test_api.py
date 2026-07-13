import http.client
import json
import os
import time

def upload_image(filepath):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(filepath, "rb") as f:
        file_data = f.read()
    filename = os.path.basename(filepath)
    body = []
    body.append(f"--{boundary}\r\n".encode())
    body.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode())
    body.append(b"Content-Type: image/jpeg\r\n\r\n")
    body.append(file_data)
    body.append(f"\r\n--{boundary}--\r\n".encode())
    data = b"".join(body)
    
    conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=120)
    conn.request("POST", "/upload", data, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    result = json.loads(resp.read())
    conn.close()
    return result

def inpaint(image_id, mask_ids):
    body = json.dumps({"image_id": image_id, "mask_ids": mask_ids})
    conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=120)
    conn.request("POST", "/inpaint", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    result = json.loads(resp.read())
    conn.close()
    return result

# Test upload
print("Testing upload...")
upload_res = upload_image("shapes.jpg")
print(f"Upload success: {upload_res.get('success')}")
print(f"Masks: {len(upload_res.get('masks', []))}")
print(f"Shape: {upload_res.get('shape')}")

if upload_res.get("success") and upload_res.get("masks"):
    mask_ids = [m["id"] for m in upload_res["masks"][:3]]
    print(f"\nTesting inpaint with masks {mask_ids}...")
    inpaint_res = inpaint(upload_res["image_id"], mask_ids)
    print(f"Inpaint success: {inpaint_res.get('success')}")
    print(f"Result length: {len(inpaint_res.get('result', ''))} chars")

print("\nDONE")
