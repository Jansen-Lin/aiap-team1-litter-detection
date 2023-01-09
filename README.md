# Litter Detection with YOLOv5 & TACO Dataset

__Contributors:__
 [Ashton](https://www.linkedin.com/in/ashton-pang-xq/) |
  [Chin Hee](https://www.linkedin.com/in/ongchinhee/) |
  [Jansen](https://www.linkedin.com/feed/) |
  [Yongquan](https://www.linkedin.com/in/yongquan-c-82aa91255/)


## Introduction
This repo was created for the final week of our deepskilling phase, <br>
where we did a mini-project to help improve AISG's workflow and welfare. <br>

![demo](https://user-images.githubusercontent.com/93126390/211184416-672dfd94-9876-4329-a74d-5e8109dac99e.jpg)

The objection detection model is trained using the open-source [YOLOv5](https://github.com/ultralytics/yolov5) project from Ultralytics while the dataset is a subset of the [TACO dataset](https://github.com/pedropro/TACO).


## Overview of Training Metrics

![metrics](https://user-images.githubusercontent.com/93126390/211184435-0c69c87c-6465-4819-98c4-e899b7325f35.jpg)
![train-loss](https://user-images.githubusercontent.com/93126390/211184438-5d7a0da5-4979-4de8-bfe6-55f6c2fdeb95.png)
![val-loss](https://user-images.githubusercontent.com/93126390/211184439-9c5c17e9-a77d-48f4-9030-a4e1127e827b.png)


## Setting up the environment

To create a new conda environment & activate it:

```
conda create -n pkd-litter python=3.8

conda activate pkd-litter
```

To install PyTorch for __windows OS users__ (currently only tested for this):
```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

To install PyTorch for __other OS users__ (not tested): <br>

- Refer to https://pytorch.org/ for the respective command to install PyTorch.


## Running the litter detection engine

Ensure you have a webcam connected to your computer.

To start the litter detection engine:
```
peekingduck run
```
