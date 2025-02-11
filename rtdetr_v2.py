import torch
import cv2
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import get_device

# Set device
device = get_device()
print(f"Using device: {device}")

# Initialize models
image_processor_r101 = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r101vd")
image_processor_r18 = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model_r101 = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r101vd").to(device)
model_r18 = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(device)

cap = cv2.VideoCapture(0)

# Initialize plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initialize timing lists
times_r101 = []
times_r18 = []

# Get model sizes in MB
def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)  # Convert bytes to MB

model_size_r101 = get_model_size_mb(model_r101)
model_size_r18 = get_model_size_mb(model_r18)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Process with R101
    start_time = time.time()
    inputs_r101 = image_processor_r101(images=pil_image, return_tensors="pt")
    inputs_r101 = {k: v.to(device) for k, v in inputs_r101.items()}
    with torch.no_grad():
        outputs_r101 = model_r101(**inputs_r101)
    results_r101 = image_processor_r101.post_process_object_detection(
        outputs_r101,
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]),
        threshold=0.5
    )
    inference_time_r101 = (time.time() - start_time) * 1000
    times_r101.append(inference_time_r101)
    
    # Process with R18
    start_time = time.time()
    inputs_r18 = image_processor_r18(images=pil_image, return_tensors="pt")
    inputs_r18 = {k: v.to(device) for k, v in inputs_r18.items()}
    with torch.no_grad():
        outputs_r18 = model_r18(**inputs_r18)
    results_r18 = image_processor_r18.post_process_object_detection(
        outputs_r18,
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]),
        threshold=0.5
    )
    inference_time_r18 = (time.time() - start_time) * 1000
    times_r18.append(inference_time_r18)
    
    # Create copies for visualization
    frame_r101 = frame.copy()
    frame_r18 = frame.copy()
    
    # Draw detections for R101
    for result in results_r101:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model_r101.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            cv2.rectangle(frame_r101, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame_r101, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw detections for R18
    for result in results_r18:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model_r18.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            cv2.rectangle(frame_r18, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame_r18, label_text, (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Calculate average times
    avg_time_r101 = sum(times_r101[-30:]) / min(len(times_r101), 30)
    avg_time_r18 = sum(times_r18[-30:]) / min(len(times_r18), 30)
    
    # Update matplotlib display
    ax1.clear()
    ax2.clear()
    ax1.imshow(cv2.cvtColor(frame_r101, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(frame_r18, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"RT-DETR R101 ({model_size_r101:.1f}MB)\nAvg: {avg_time_r101:.1f}ms")
    ax2.set_title(f"RT-DETR R18 ({model_size_r18:.1f}MB)\nAvg: {avg_time_r18:.1f}ms")
    ax1.axis('off')
    ax2.axis('off')
    plt.pause(0.01)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
