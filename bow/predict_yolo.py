from ultralytics import YOLO

# Indlæs den bedste vægtfil
model = YOLO("runs/detect/train/weights/best.pt")

# Test modellen på testbillederne
results = model.predict(
    source="C:/Users/Grams/OneDrive/Skrivebord/projectBow/ProjectBow.v1i.yolov11/test/images",
    save=True
)

# Print resultater
print(results)
