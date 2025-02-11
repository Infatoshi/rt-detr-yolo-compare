import torch
import cv2
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import numpy as np
from utils import get_device

# Set device
device = get_device()
print(f"Using device: {device}")

cap = cv2.VideoCapture(0)

image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(device)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    inputs = image_processor(images=pil_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
        threshold=0.5
    )
    
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score = score.item()
            label = model.config.id2label[label_id.item()]
            box = [int(i) for i in box.tolist()]
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (box[0], box[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
