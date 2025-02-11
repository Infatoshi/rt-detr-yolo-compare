from ultralytics import YOLO
import cv2
import time
import matplotlib.pyplot as plt
from utils import get_device
import os

def main():
    # Load both YOLO models
    model_n_path = 'yolo11n.pt'
    model_x_path = 'yolo11x.pt'
    model_n = YOLO(model_n_path)
    model_x = YOLO(model_x_path)
    
    # Get model file sizes
    model_n_size = f"{os.path.getsize(model_n_path) / (1024*1024):.1f}MB"
    model_x_size = f"{os.path.getsize(model_x_path) / (1024*1024):.1f}MB"
    
    device = get_device()
    model_n.to(device)
    model_x.to(device)
    
    # Initialize webcam and plotting
    cap = cv2.VideoCapture(0)
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize timing lists
    times_n = []
    times_x = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Run YOLOv11n inference and measure time
        start_time = time.time()
        results_n = model_n(frame, device=device)
        inference_time_n = (time.time() - start_time) * 1000  # Convert to ms
        times_n.append(inference_time_n)
        
        # Run YOLOv11x inference and measure time
        start_time = time.time()
        results_x = model_x(frame, device=device)
        inference_time_x = (time.time() - start_time) * 1000  # Convert to ms
        times_x.append(inference_time_x)
        
        # Get annotated frames
        frame_n = results_n[0].plot()
        frame_x = results_x[0].plot()
        
        # Calculate average times
        avg_time_n = sum(times_n[-30:]) / min(len(times_n), 30)
        avg_time_x = sum(times_x[-30:]) / min(len(times_x), 30)
        
        # Add timing and size information to frames
        cv2.putText(frame_n, f"YOLOv11n ({model_n_size}) - {avg_time_n:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_x, f"YOLOv11x ({model_x_size}) - {avg_time_x:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Update matplotlib display
        ax1.clear()
        ax2.clear()
        ax1.imshow(cv2.cvtColor(frame_n, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(frame_x, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"YOLOv11n (Nano - {model_n_size})\nAvg: {avg_time_n:.1f}ms")
        ax2.set_title(f"YOLOv11x (XLarge - {model_x_size})\nAvg: {avg_time_x:.1f}ms")
        ax1.axis('off')
        ax2.axis('off')
        plt.pause(0.01)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()