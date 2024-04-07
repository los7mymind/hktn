# hktn

## Kaggle

```python
!pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO(<model path>)

results = model.train(
   data='<data yaml path>',
   imgsz=640,
   epochs=50,
   batch=32, # изначально подавали по 8, но потыкались и 32 оказался продуктивнее всех
   name='hktn')
```
