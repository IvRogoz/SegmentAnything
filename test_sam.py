from ultralytics import SAM
model = SAM("sam2_t.pt")
results = model("test.jpg", device="cuda", verbose=True)
if results[0].masks:
    print(f"Masks: {len(results[0].masks)}")
else:
    print("No masks")
