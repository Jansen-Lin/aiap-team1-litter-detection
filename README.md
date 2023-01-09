# Litter Detection with YOLOv5 & TACO Dataset

__Contributors:__
 [Ashton](https://www.linkedin.com/in/ashton-pang-xq/) |
  [Chin Hee](https://www.linkedin.com/in/ongchinhee/) |
  [Jansen](https://www.linkedin.com/feed/) |
  [Yongquan](https://www.linkedin.com/in/yongquan-c-82aa91255/)

This repo was created for the final week of the deep-skilling phase as part of our
__[AI apprenticeship programme (AIAP)](https://aisingapore.org/innovation/aiap/)__, where we did a mini-project to help improve AISG's workflow and welfare. 

## Introduction

We envision having a moving robot around the office to detect litters, like the example shown below, where a plastic thrash is detected with high confidence of 98%.

![demo](https://user-images.githubusercontent.com/93126390/211184416-672dfd94-9876-4329-a74d-5e8109dac99e.jpg)

This is made possible via an objection detection model trained using the open-source [YOLOv5](https://github.com/ultralytics/yolov5) project from Ultralytics, on a subset of the [TACO dataset](https://github.com/pedropro/TACO).


## Overview of Training & Evaluation Metrics

This section gives an overview of how the various training & evaluation metrics changed with an increase in epoch. Two experiments (exp3, exp4) are shown here.

![metrics](https://user-images.githubusercontent.com/93126390/211184435-0c69c87c-6465-4819-98c4-e899b7325f35.jpg)
![train-loss](https://user-images.githubusercontent.com/93126390/211184438-5d7a0da5-4979-4de8-bfe6-55f6c2fdeb95.png)
![val-loss](https://user-images.githubusercontent.com/93126390/211184439-9c5c17e9-a77d-48f4-9030-a4e1127e827b.png)

- exp4 gave the higher final evaluation scores (mAP_0.5, mAP_0.5:0.95, precision, recall), thus the resulting model weight (found in `model/yolov5n_taco_best.pt`) is used for inference, and incorporating part of the codebase from [PeekDuck](https://github.com/aisingapore/PeekingDuck)
- Both training and validation losses looks to be still decreasing, which suggest perhaps more epoch can be done for even better results.

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
Now a new window pops up which shows where your webcam is pointed at and also detects litters.

To end the session, just do a `CTRL C` on the command line, or simply close the pop-up window.

## Files Output

For each session, two items will be created in `processed/` folder:
- CSV file
- mp4 file

## View Files Output on Streamlit

Work in Progress...