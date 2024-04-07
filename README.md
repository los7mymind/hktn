# hktn

в качестве виртуальной машины использовали мощности Kaggle. для сбора аналитики использовали API Weights & Basis (wandb.ai).

изначально пробовали организовать работу через Roboflow и ClearML, но посчитали решения выше более удобными.

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
