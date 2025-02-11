import torch
import cv2
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import get_device
import time

# Set device
device = get_device()
print(f"Using device: {device}")

# Initialize models
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r101vd")
rtdetr_model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r101vd").to(device)
rtdetr_r18_model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(device)
yolo_n_model = YOLO('yolo11n.pt')
yolo_x_model = YOLO('yolo11x.pt')
yolo_n_model.to(device)
yolo_x_model.to(device)

# Get model sizes
def get_model_size_mb(model):
    if hasattr(model, 'state_dict'):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    else:
        return model.model.state_dict().get('model.0.conv.weight').numel() * 4 / (1024 * 1024)

rtdetr_size = get_model_size_mb(rtdetr_model)
rtdetr_r18_size = get_model_size_mb(rtdetr_r18_model)
yolo_n_size = get_model_size_mb(yolo_n_model)
yolo_x_size = get_model_size_mb(yolo_x_model)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up matplotlib figure
plt.ion()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.canvas.manager.set_window_title('Object Detection Comparison')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame for RTDetr
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # RTDetr detection
    start_time = time.time()
    inputs = image_processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = rtdetr_model(**inputs)
    
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
        threshold=0.5
    )
    rtdetr_time = (time.time() - start_time) * 1000  # Convert to ms
    
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
    
    # RTDetr R18 detection
    start_time = time.time()
    inputs = image_processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = rtdetr_r18_model(**inputs)
    
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
        threshold=0.5
    )
    rtdetr_r18_time = (time.time() - start_time) * 1000  # Convert to ms
    
    rtdetr_r18_frame = frame.copy()
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = rtdetr_r18_model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            
            cv2.rectangle(rtdetr_r18_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(rtdetr_r18_frame, label_text, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # YOLO-n detection
    start_time = time.time()
    yolo_n_results = yolo_n_model(frame)
    yolo_n_time = (time.time() - start_time) * 1000
    yolo_n_frame = yolo_n_results[0].plot()
    
    # YOLO-x detection
    start_time = time.time()
    yolo_x_results = yolo_x_model(frame)
    yolo_x_time = (time.time() - start_time) * 1000
    yolo_x_frame = yolo_x_results[0].plot()
    
    # Convert BGR to RGB for matplotlib
    rtdetr_frame_rgb = cv2.cvtColor(rtdetr_frame, cv2.COLOR_BGR2RGB)
    yolo_n_frame_rgb = cv2.cvtColor(yolo_n_frame, cv2.COLOR_BGR2RGB)
    yolo_x_frame_rgb = cv2.cvtColor(yolo_x_frame, cv2.COLOR_BGR2RGB)
    rtdetr_r18_frame_rgb = cv2.cvtColor(rtdetr_r18_frame, cv2.COLOR_BGR2RGB)
    
    # Clear previous plots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.clear()
    
    # Display frames with timing and model size information
    ax1.imshow(yolo_n_frame_rgb)
    ax1.text(1.02, 0.5, f'YOLOv11-n Detection\n{yolo_n_time:.1f}ms\n{yolo_n_size:.1f}MB\nFPS: {1000/yolo_n_time:.1f}',
             transform=ax1.transAxes, verticalalignment='center')
    ax1.axis('off')
    
    ax2.imshow(rtdetr_frame_rgb)
    ax2.text(1.02, 0.5, f'RTDetr-R101 Detection\n{rtdetr_time:.1f}ms\n{rtdetr_size:.1f}MB\nFPS: {1000/rtdetr_time:.1f}',
             transform=ax2.transAxes, verticalalignment='center')
    ax2.axis('off')
    
    ax3.imshow(yolo_x_frame_rgb)
    ax3.text(1.02, 0.5, f'YOLOv11-x Detection\n{yolo_x_time:.1f}ms\n{yolo_x_size:.1f}MB\nFPS: {1000/yolo_x_time:.1f}',
             transform=ax3.transAxes, verticalalignment='center')
    ax3.axis('off')
    
    ax4.imshow(rtdetr_r18_frame_rgb)
    ax4.text(1.02, 0.5, f'RTDetr-R18 Detection\n{rtdetr_r18_time:.1f}ms\n{rtdetr_r18_size:.1f}MB\nFPS: {1000/rtdetr_r18_time:.1f}',
             transform=ax4.transAxes, verticalalignment='center')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.pause(0.001)  # Small pause to update the plot
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.ioff()
plt.close()
