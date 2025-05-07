from ultralytics import YOLO
import wandb
wandb.init(project='YOLOv8')
#for epoch in range(200):
#   wandb.log({"metrics/precision(B)",
#              "metrics/recall(B)",
#              "metrics/mAP50(B)",
#               "metrics/mAP50-95(B)",
#               "time",
#               "train/box_loss",
#               "train/cls_loss",
#               "train/dfl_loss"
#               },step = epoch)
wandb.login(key="2cf727e31b6163d16c9a162a26d2ac4cce99bf76")

def main():
    data_path = '/home/hoo/CRY/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml'
    model = YOLO('/home/hoo/CRY/ultralytics-main/ultralytics/cfg/models/v8/myyolov8m-GhostConv-SE.yaml')
    model.load('yolov8m.pt')

    model.train(
        data=data_path,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
    )

if __name__ =="__main__":
    main()

