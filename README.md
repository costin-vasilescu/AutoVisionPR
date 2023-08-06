# AutoVisionPR

Demonstration: https://www.youtube.com/watch?v=vR95KOq2xaI

This project aims to detect in real-time romanian vehicle license plates and extract the text from them using OCR.

Objectives:
- create a proper dataset using real and synthetic generated license plates
- train a YOLOv8 model for object detection
- detect and track license plates using YOLOv8 and BoT-SORT
- extract text using LanyOCR
- evaluate and update stored detections based on a scoring system to only keep the most accurate ones

Technologies used: <br />
Albumentations - https://github.com/albumentations-team/albumentations <br />
YOLOv8 - https://github.com/ultralytics/ultralytics <br />
BoT-SORT - https://github.com/NirAharon/BoT-SORT <br />
LanyOCR - https://github.com/JC1DA/lanyocr
