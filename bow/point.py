from ultralytics import YOLO

# Load din model
model = YOLO("path/to/your-model.pt")

# Analyser billeder og gem resultaterne
results = model.predict(source="path/to/test/images", save=True)

# Definer pointtildeling baseret på farve
point_system = {
    "Arrow_Yellow": 10,
    "Arrow_Red": 8,
    "Arrow_Blue": 6,
    "Arrow_Black": 4,
    "Arrow_White": 2
}

# Analyser resultater og beregn point
total_points = 0
for result in results:
    print(f"Processing image: {result.path}")
    for box in result.boxes:
        class_id = int(box.cls[0])  # Klassens ID
        class_name = result.names[class_id]  # Klassens navn
        points = point_system.get(class_name, 0)
        total_points += points
        print(f"Detected: {class_name}, Points: {points}")

print(f"Total points: {total_points}")
