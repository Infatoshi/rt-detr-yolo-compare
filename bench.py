import torch
import cv2
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import get_device

# Set device
device = get_device()
print(f"Using device: {device}")

# Initialize models
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r101vd")
rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r101vd").to(device)
yolo_model = YOLO('yolov11n.pt')
yolo_model.to(device)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up matplotlib figure
plt.ion()  # Enable interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.canvas.manager.set_window_title('Object Detection Comparison')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame for RTDetr
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # RTDetr detection
    inputs = image_processor(images=pil_image, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = rtdetr_model(**inputs)
    
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
        threshold=0.5
    )
    
    rtdetr_frame = frame.copy()
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = rtdetr_model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            
            cv2.rectangle(rtdetr_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(rtdetr_frame, label_text, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # YOLO detection
    yolo_results = yolo_model(frame)
    yolo_frame = yolo_results[0].plot()
    
    # Convert BGR to RGB for matplotlib
    rtdetr_frame_rgb = cv2.cvtColor(rtdetr_frame, cv2.COLOR_BGR2RGB)
    yolo_frame_rgb = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2RGB)
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Display frames
    ax1.imshow(rtdetr_frame_rgb)
    ax1.set_title('RTDetr Detection')
    ax1.axis('off')
    
    ax2.imshow(yolo_frame_rgb)
    ax2.set_title('YOLO Detection')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.pause(0.001)  # Small pause to update the plot
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.ioff()
plt.close()
