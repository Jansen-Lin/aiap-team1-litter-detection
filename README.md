# Litter Detection with YOLOv5 & TACO Dataset

This repo is created for the final week of the deepskilling phase, <br>
where we do a mini-project to help improve AISG's workflow and welfare. <br>

<p>
    <img src="./metrics/demo.jpg" />
</p>

The model is trained using the open-source [YOLOv5](https://github.com/ultralytics/yolov5) project from Ultralytics while the dataset is a subset of the [TACO dataset](https://github.com/pedropro/TACO).

# Training Metrics
![Metrics](./metrics/metrics.jpg)
![Metrics](./metrics/train_loss.png)
![Metrics](./metrics/val_loss.png)

Installation
------------
```
> conda create -n pkd python=3.8
> conda activate pkd
> pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
> peekingduck run
```
