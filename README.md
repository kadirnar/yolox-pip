<div align="center">
<h3>
  Yolox-Pip: This is a packaged version of the YOLOX for easy installation and use.
</h3>
<h4>
    <img width="800" alt="teaser" src="doc/fig.png">
</h4>
</div>

## <div align="center">Overview</div>

This repo is a packaged version of the [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for easy installation and use.
### Installation
```
pip install yoloxdetect
```

### Yolox Inference
```python
from yoloxdetect import YoloxDetector

model = YoloxDetector(
    model_path = "data/weights/yolox_s.pth",
    config_path = "configs.yolox_s",
    device = "cuda:0",
)
model.classes = None
model.conf = 0.25
model.iou = 0.45
model.show = False
model.save = True

pred = model.predict(image='data/images', img_size=640)
```
### Citation
```bibtex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
