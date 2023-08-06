# AutoVisionPR

Bachelor thesis project aiming to detect Romanian license plates in real-time using computer vision. <br />
Demo: https://www.youtube.com/watch?v=vR95KOq2xaI

Objectives:
- create a proper dataset using real and synthetic generated license plates
- train a YOLOv8 model for object detection
- detect and track license plates using YOLOv8 and BoT-SORT
- extract text using LanyOCR
- evaluate and update stored detections using a scoring system to only keep the most accurate detections
- store detected information in an SQLite database using a FastAPI backend
- put everything together in a Streamlit application with useful features

Technologies used: <br />
OpenCV - https://github.com/opencv/opencv-python <br />
Pillow - https://github.com/python-pillow/Pillow <br />
Albumentations - https://github.com/albumentations-team/albumentations <br />
YOLOv8 - https://github.com/ultralytics/ultralytics <br />
BoT-SORT - https://github.com/NirAharon/BoT-SORT <br />
LanyOCR - https://github.com/JC1DA/lanyocr <br />
FastAPI - https://github.com/tiangolo/fastapi <br />
SQLAlchemy - https://github.com/sqlalchemy/sqlalchemy <br />
Streamlit - https://github.com/streamlit/streamlit <br />

