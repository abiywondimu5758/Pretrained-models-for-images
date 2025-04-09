import cv2
import torch
import time
from torchvision import models, transforms

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer to measure inference time
    start_inference = time.time()
    
    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the transformation to the frame
    input_tensor = transform(rgb_frame)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Measure inference time
    inference_time = time.time() - start_inference

    # Initialize metrics for the current frame
    detections = 0
    confidence_sum = 0.0

    # Process predictions
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.5:  # Confidence threshold
            detections += 1
            confidence_sum += score.item()
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Label: {label}, Score: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Compute average confidence if there are detections
    avg_confidence = confidence_sum / detections if detections > 0 else 0

    # Calculate FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Overlay metrics on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Inference Time: {inference_time*1000:.1f}ms', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f'Detections: {detections}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Avg Confidence: {avg_confidence:.2f}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display the frame with detection info
    cv2.imshow('Faster R-CNN Real-Time Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()