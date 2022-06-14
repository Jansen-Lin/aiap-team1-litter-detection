# Litter Detection with YOLOv5 & TACO Dataset

This repo is created for the final week of the deepskilling phase, <br>
where we do a mini-project to help improve AISG's workflow and welfare. <br>

![Demo](https://gitlab.aisingapore.net/aiap/deep-skilling-phase/aiap10/team1/-/raw/yongquan_chen/metrics/demo.jpg)

The model is trained using the open-source [YOLOv5](https://github.com/ultralytics/yolov5) project from Ultralytics while the dataset is a subset of the [TACO dataset](https://github.com/pedropro/TACO).

# Training Metrics
![Metrics](/uploads/889ca5ac7756441ffd00d6de88a1e070/metrics.png)
![Metrics](/uploads/dc60a2a7c780abd29c676e2f525d99f9/train_loss.png)
![Metrics](/uploads/ca61ac8ba178812266e83e238c07fbd0/val_loss.png)

Installation
------------
```
> conda create -n pkd python=3.8
> conda activate pkd
> pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
> peekingduck run
```
