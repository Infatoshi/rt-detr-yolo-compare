from ultralytics import YOLO
import cv2
import time
from utils import get_device

def main():
    # Load YOLO model
    model = YOLO('yolov11n.pt')
    device = get_device()
    model.to(device)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Run YOLOv11 inference on the frame
        results = model(frame, device=device)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display device info
        cv2.putText(annotated_frame, f"Device: {device}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Press 'q' to quit", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the annotated frame
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()