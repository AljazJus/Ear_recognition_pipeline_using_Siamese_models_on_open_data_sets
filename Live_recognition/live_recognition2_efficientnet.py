from ultralytics import YOLO
import cv2
import math 
from torchvision.ops import nms
import torch
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from Siamese_efficientNet import SiameseNetwork
from Unified_model import Recognition 
import Images2Vectors as iv
from torchvision.models import efficientnet_b0
import torch.nn as nn
import time
# model
yolo_model = YOLO("/path/to/yolo/weights.pt")


model_path="/path/to/siamese/model.pth"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
# Load state_dict into classifier
model_base = efficientnet_b0()
model_base.classifier = nn.Identity()
model_base.load_state_dict(checkpoint['model_state_dict'])
model_base.eval()
siamses_model= SiameseNetwork(model_base)
# print(model)
siamses_model.classifier.load_state_dict(checkpoint['linear_state_dict'])
_=siamses_model.eval()


embeddings_path = "path/to/embeddings"

# try:
#     names,ears_tensors= iv.load_embeddings_from_file(embeddings_path)
# except:
images="/path/to/images"
names="path/to/names.csv"
ears_tensors, names = iv.images2vectors(images,names,siamses_model)
iv.save_embeddings_to_file(ears_tensors,names, embeddings_path)


recognition_model=Recognition(siamses_model,yolo_model,(ears_tensors,names))
# object classes


# start webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)

time_tables = {}

while True:
    success, img = cap.read()

    start = time.time()
    results, (boxes, labels) = recognition_model(img, stream=True)
    end = time.time()

    if len(labels) != 0:
        # Initialize a new list if this is a new key
        if len(labels) not in time_tables:
            time_tables[len(labels)] = []
        time_tables[len(labels)].append(end - start)
    # print(labels)
    # Plot the remaining boxes
    for box,label in zip(boxes,labels):
        x1, y1, x2, y2 =box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        
        org = [x1, y1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        
        if label[0] == -1:
            cv2.putText(img, "Unknown", org, font, fontScale, color, thickness)
            print(f"Unknown: {label[1]}")
        else :
            cv2.putText(img, str(label[0]), org, font, fontScale, color, thickness)
            print(f"{label[0]}: {label[1]}")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("Average time for each class:")
for key in time_tables:
    print(f"{key}: {sum(time_tables[key]) / len(time_tables[key])}")