import cv2
import torch
import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initial time and frame counter for FPS measurement
prev_time = time.time()
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Perform inference and measure inference time
    inf_start = time.time()
    results = model(frame)
    inf_end = time.time()
    inference_time = (inf_end - inf_start) * 1000  # Convert to milliseconds

    # Compute detection count and average confidence
    detections = results.xyxy[0]  # tensor with [x1, y1, x2, y2, conf, class]
    detection_count = detections.shape[0]
    avg_confidence = detections[:, 4].mean().item() if detection_count > 0 else 0

    # Render results on the frame
    frame = results.render()[0].copy()  # Make a writable copy of the rendered frame

    # Overlay metrics on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Inference Time: {inference_time:.2f} ms', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Detections: {detection_count}', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Avg Conf: {avg_confidence:.2f}', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()