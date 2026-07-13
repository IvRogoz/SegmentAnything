import http.client, json
boundary = "testboundary"
with open("shapes.jpg", "rb") as f:
    data = f.read()
body = b"--" + boundary.encode() + b'\r\nContent-Disposition: form-data; name="image"; filename="sh.jpg"\r\nContent-Type: image/jpeg\r\n\r\n' + data + b"\r\n--" + boundary.encode() + b"--\r\n"
conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=120)
conn.request("POST", "/upload", body, {"Content-Type": "multipart/form-data; boundary=" + boundary})
resp = conn.getresponse()
print(f"Status: {resp.status}")
raw = resp.read()
print(raw.decode("utf-8", errors="replace"))
