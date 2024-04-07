# hktn

## Kaggle

```python
!pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO('/kaggle/working/runs/detect/yolov8n_custom4/weights/last.pt')

results = model.train(
   data='/kaggle/input/mainmainmain/1/data.yaml',
   imgsz=640,
   epochs=50,
   batch=32,
   name='yolov8n_custom')
```
