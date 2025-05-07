from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("/home/hoo/CRY/ultralytics-main/runs/detect/train57/weights/best.pt")

image_path = "ultralytics/assets/2.jpg"
image = cv2.imread(image_path)

result = model.predict(source=image_path, save=True, conf=0.25)

annotated_image = results[0].plot()
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

result.save(save_dir="ultralytics/runs/test")