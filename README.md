# Real-Time Object Detection Demo

This demo uses a pretrained Faster R-CNN (ResNet-50 FPN) model from PyTorch’s `torchvision` to perform real-time object detection via your webcam.

## Prerequisites

- Python 3.6 or later  
- PyTorch 1.7+ and torchvision matched to your CUDA/CPU setup  
- OpenCV (`opencv-python`)

## Installation

1. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install torch torchvision opencv-python
   ```

## Usage

1. Navigate to this folder.
2. Run the detection script:
   ```
   python resnet.py
   ```
3. A window titled **Faster R-CNN Real-Time Detection** will open.  
4. Press **q** to quit.

## Notes

- You can adjust the confidence threshold in `resnet.py` (default 0.5).  
- To switch cameras, change the index in `cv2.VideoCapture(0)`.

## License

MIT License  
