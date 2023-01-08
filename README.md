# Litter Detection with YOLOv5 & TACO Dataset

This repo is created for the final week of the deepskilling phase, <br>
where we do a mini-project to help improve AISG's workflow and welfare. <br>

![demo](https://user-images.githubusercontent.com/93126390/211184416-672dfd94-9876-4329-a74d-5e8109dac99e.jpg)

The model is trained using the open-source [YOLOv5](https://github.com/ultralytics/yolov5) project from Ultralytics while the dataset is a subset of the [TACO dataset](https://github.com/pedropro/TACO).

# Training Metrics

![metrics](https://user-images.githubusercontent.com/93126390/211184435-0c69c87c-6465-4819-98c4-e899b7325f35.jpg)
![train-loss](https://user-images.githubusercontent.com/93126390/211184438-5d7a0da5-4979-4de8-bfe6-55f6c2fdeb95.png)
![val-loss](https://user-images.githubusercontent.com/93126390/211184439-9c5c17e9-a77d-48f4-9030-a4e1127e827b.png)

Installation
------------
```
> conda create -n pkd python=3.8
> conda activate pkd
> pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
> peekingduck run
```
