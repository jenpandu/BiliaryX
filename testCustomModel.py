from ultralytics import YOLO

# Load your trained model
model = YOLO("custom_yolov8n_biliary.pt")  # use the path to your trained model

# Run prediction on a single image
results = model.predict(
    source="trainImage.png",  # replace with your image path
    conf=0.25,  # confidence threshold
    save=True,  # saves the results to runs/detect/predict
    show=True   # displays the image with predictions
)

# Print the detection results
for result in results:
    boxes = result.boxes  # get boxes on the image
    for box in boxes:
        print(f"Confidence: {box.conf.item():.2f}")
        print(f"Coordinates: {box.xyxy.tolist()}")  # box coordinates

