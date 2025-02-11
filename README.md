![Image](assets/fluffy-box.jpg)

# Real-time Object Detection Comparison

This project provides a side-by-side comparison of two popular object detection models: RTDetr and YOLOv8, running in real-time using your webcam.

## Features

- Real-time object detection using webcam input
- Side-by-side comparison of RTDetr and YOLOv8 models
- Interactive visualization using matplotlib
- Support for CPU and MPS (Apple Silicon) devices

## Prerequisites

- Python 3.8 or higher
- Webcam

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Infatoshi/rt-detr-yolo-compare
cd rt-detr-yolo-compare
```

2. Install required packages:
```bash
pip install git+https://github.com/huggingface/transformers torch opencv-python Pillow numpy matplotlib ultralytics
```

3. Download the yolov11n.pt and 11x.pt models:
```bash
yolo predict model=yolo11x.pt source='assets/fluffy.png' && yolo predict model=yolo11n.pt source='assets/fluffy.png'
```

## Usage

The project includes three main scripts:

1. bench.py - Runs both models side-by-side for comparison:
   python bench.py

2. yolo.py - Runs only YOLOv8 detection:
   python yolo.py

3. rtdetr_v2.py - Runs only RTDetr detection:
   python rtdetr_v2.py

Press `q` to quit any of the running scripts.

Also... for a one liner yolo v11 usage. Try 
```bash
yolo predict model=yolo11x.pt source='assets/fluffy.png'
```

## Models Used

- RTDetr_V2_R101: `PekingU/rtdetr_v2_r101vd` (resnet101 base)
- RTDetr_V2_R18: `PekingU/rtdetr_v2_r18vd` (resnet18 base)
- YOLOv11_X: `yolo11x.pt` (xlarge version)
- YOLOv11_N: `yolo11n.pt` (nano version)


## Performance Notes

- The script automatically detects and uses CUDA or MPS (Apple Silicon - Metal Performance Shaders) if available, otherwise falls back to CPU
- RTDetr_V2_R101 -> more accuracy, less speed
- RTDetr_V2_R18 -> less accuracy, more speed
- YOLOv11_X -> more accuracy, less speed
- YOLOv11_N -> less accuracy, more speed

## Troubleshooting

If you encounter any issues:

1. Ensure your webcam is properly connected and accessible
2. Verify all dependencies are correctly installed
3. Make sure you have proper permissions to access the webcam
