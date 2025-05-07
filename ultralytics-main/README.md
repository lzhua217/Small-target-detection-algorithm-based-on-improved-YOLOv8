# Small target detection algorithm based on improved YOLOv8 for UAV aerial images scene

This repo contains the code for the Small-target-detection-algorithm-based-on-improved-YOLOv8, the original paper is:

> **Small target detection algorithm based on improved YOLOv8 for UAV aerial images scene**<br>
> Ruiyun Chen, Zhonghua Liu, Qiang Zhao, Weihua Ou, Kaibing Zhang<br>
> \[[Paper](...)\]

## To get started

### 1. Requirements

Run `pip install ultralytics` in terminal.

### 2. Prepare Visdrone-2019 dataset 

(a) You can download the dataset from https://github.com/VisDrone/VisDrone-Dataset#task-1-object-detection-in-images.  

(b) Convert data form to Yolo by running `visDrone2yolov.py` (you may need to change the `dir`).  

We suppose the data directory is constructed as
```
Your project name
├── datasets
|   ├── VisDrone2019
|   |   └── VisDrone2019-DET-train
            └── annotations
            └── images
            └── labels
|   |   └── VisDrone2019-DET-val
            └── annotations
            └── images
            └── labels
|   |   └── VisDrone2019-DET-test-dev
            └── annotations
            └── images
            └── labels
├── ultralytics-main
```
(c) Modify path args in `data/VisDrone.yaml` . 

### 3. Train the model

Modify args in `train.py`. In ultralytics-main, some args are set as follows:

* `--weights`: `yolov8m.pt`
* `-model`: `models/myyolov8m-GhostConv-ContextAggregation-WIoU.yaml`
* `--data_path`: `ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml`

### 4. Evaluate the model

Modify args in `val.py`.

* `--weights`: the `best.pt` file in your result folder.
* `--task`: can be `val` or `test`.


## Contact

My email: 2380779262@qq.com
