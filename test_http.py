import http.client, json
boundary = "testboundary"
with open("shapes.jpg", "rb") as f:
    data = f.read()
print(f"File size: {len(data)}")
body = b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="image"; filename="sh.jpg"\r\nContent-Type: image/jpeg\r\n\r\n' + data + b"\r\n--" + boundary.encode() + b"--\r\n"
conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=120)
conn.request("POST", "/upload", body, {"Content-Type": "multipart/form-data; boundary=" + boundary})
resp = conn.getresponse()
print(f"Status: {resp.status}")
raw = resp.read()
print(f"Raw len: {len(raw)}")
try:
    r = json.loads(raw)
    print("Has background:", "background" in r)
    print("Bg len:", len(r.get("background", "")))
    print("Masks:", len(r.get("masks", [])))
except Exception as e:
    print(f"JSON error: {e}")
    print("First 500:", raw[:500].decode("utf-8", errors="replace"))
