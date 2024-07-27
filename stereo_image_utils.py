# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

# Helper functions
def preprocess_image(image):
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image

def get_detections(maskrcnn, img, score_threshold=0.5):
    with torch.no_grad():
        result = maskrcnn(preprocess_image(img))[0]

    mask = result["scores"] > score_threshold

    boxes = result["boxes"][mask].detach().cpu().numpy()
    labels = result["labels"][mask].detach().cpu().numpy()
    scores = result["scores"][mask].detach().cpu().numpy()
    masks = result["masks"][mask]

    return boxes, labels, scores, masks

def annotate_frame(frame, boxes, labels, scores, distances):
    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i])
        score = scores[i]
        dist = distances[i]
        
        tlx, tly, brx, bry = map(int, box)
        cv2.rectangle(frame, (tlx, tly), (brx, bry), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {score:.2f} {dist:.2f} cm', (tlx, tly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Initialize model
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = get_detections(model, img)
    
    boxes, labels, scores, masks = detections
    
    # Calculate distances based on detections (placeholder logic)
    distances = [0] * len(boxes)  # Replace with your distance calculation logic

    # Annotate frame with detections and distances
    annotated_frame = annotate_frame(frame, boxes, labels, scores, distances)

    # Display the frame
    cv2.imshow('Video Feed', annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
