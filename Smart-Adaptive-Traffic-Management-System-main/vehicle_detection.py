import cv2
import numpy as np
import os

# Absolute paths to the YOLO configuration, weights, and COCO names files
cfg_path = "C:/Users/AASTIK DARYAL/Desktop/Smart-Adaptive-Traffic-Management-System-main/yolov3.cfg"
weights_path = "C:/Users/AASTIK DARYAL/Desktop/Smart-Adaptive-Traffic-Management-System-main/yolov3.weights"
names_path = "C:/Users/AASTIK DARYAL/Desktop/Smart-Adaptive-Traffic-Management-System-main/coco.names"
image_path = "C:/Users/AASTIK DARYAL/Desktop/Smart-Adaptive-Traffic-Management-System-main/testimage.jpg"  # Replace this with the actual image path

# Verify if paths are correct
for path in [cfg_path, weights_path, names_path, image_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load YOLO pre-trained model for vehicle detection
net = cv2.dnn.readNet(weights_path, cfg_path)

# Load the COCO names file (contains class names)
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Load the image
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Could not read the image: {image_path}")

# Get image dimensions
height, width, _ = image.shape

# Preprocess the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set input blob for the network
net.setInput(blob)

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Perform forward pass and get detections
detections = net.forward(output_layers)

# Initialize lists to store bounding box coordinates and confidence scores
boxes = []
confidences = []

# Loop through detections
for detection in detections:
    for obj in detection:
        scores = obj[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and class_id == 2:  # Class ID for cars is 2
            center_x = int(obj[0] * width)
            center_y = int(obj[1] * height)
            w = int(obj[2] * width)
            h = int(obj[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Apply Non-Maximum Suppression
conf_threshold = 0.5  # Confidence threshold
nms_threshold = 0.4   # NMS threshold
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Initialize vehicle count
vehicle_count = 0

# Draw bounding boxes for selected detections after NMS
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.putText(image, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        vehicle_count += 1

# Determine red and green light durations based on vehicle count
if vehicle_count <= 5:
    red_light_duration = 10
    green_light_duration = 15
elif vehicle_count <= 10:
    red_light_duration = 20
    green_light_duration = 15
elif vehicle_count <= 15:
    red_light_duration = 30
    green_light_duration = 20
elif vehicle_count <= 20:
    red_light_duration = 35
    green_light_duration = 15
elif vehicle_count <= 25:
    red_light_duration = 40
    green_light_duration = 20
elif vehicle_count <= 30:
    red_light_duration = 45
    green_light_duration = 25
elif vehicle_count <= 35:
    red_light_duration = 50
    green_light_duration = 30
elif vehicle_count <= 40:
    red_light_duration = 55
    green_light_duration = 35
elif vehicle_count <= 45:
    red_light_duration = 60
    green_light_duration = 40
elif vehicle_count <= 50:
    red_light_duration = 70
    green_light_duration = 45
else:
    red_light_duration = 65
    green_light_duration = 5


# Display the image with detections
cv2.imshow("Vehicle Detection", image)
print(f"Number of vehicles detected: {vehicle_count}")
print(f"Red light duration: {red_light_duration} seconds")
print(f"Green light duration: {green_light_duration} seconds")
cv2.waitKey(0)
cv2.destroyAllWindows()
