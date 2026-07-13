import http.client
import json
import os

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

res = upload_image("shapes.jpg")
print(f"Success: {res.get('success')}")
print(f"Masks: {len(res.get('masks', []))}")
for m in res['masks']:
    print(f"  id={m['id']} area={m['area']} score={m['score']} object_len={len(m['object'])}")
print("DONE")
