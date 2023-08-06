from ultralytics import YOLO
from lanyocr import LanyOcr
from queue import LifoQueue
from utils import match_plate_text
from datetime import datetime
import threading
import re
import numpy as np
import torch
import cv2


class OCRThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self._stop_event = threading.Event()
        self.parent = parent

    def run(self):
        while not self._stop_event.is_set():
            try:
                plate_id, plate_image = self.parent.ocr_queue.get(timeout=5)
            except:
                continue

            if plate_id is None or plate_image.size == 0:
                continue
            ocr_result = self.parent.ocr.infer(plate_image)

            best_score = self.parent.ocr_dict.get(plate_id, (None, 0))[1]
            for result in ocr_result:
                text = re.sub(r"[^a-zA-Z0-9]", "", result.text)
                score = match_plate_text(text) if len(text) >= 2 else 0
                plate_score = np.average([score, result.prob], weights=[0.7, 0.3])

                if plate_score >= best_score:
                    self.parent.ocr_dict[plate_id] = (text, plate_score)
                    best_score = plate_score

    def stop(self):
        self._stop_event.set()

    def is_stop_set(self):
        return self._stop_event.is_set()


class InferencePipeline:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = None
    ocr = None
    ocr_queue = None
    ocr_dict = None
    detections_dict = None
    results = None
    yolo_loaded = False
    lanyocr_loaded = False
    detection_counter = 0

    # Config
    conf_threshold = 0
    tracker = 'botsort.yaml'

    def load_yolo(self, model_name):
        self.model = YOLO(f'./weights/{model_name}')
        self.yolo_loaded = True

    def load_lanyocr(self):
        self.ocr = LanyOcr(
            detector_name='easyocr_craft',
            recognizer_name='paddleocr_en_ppocr_v3',
            merger_name='lanyocr_craftbased',
            merge_boxes_inference=False,
            merge_rotated_boxes=True,
            merge_vertical_boxes=True,
            use_gpu=False if self.device == 'cpu' else True,
            disable_angle_classifier=True,
            debug=False
        )
        self.lanyocr_loaded = True

    def process_frame(self, result):
        annotated_frame = result.orig_img.copy()
        for i in range(len(result)):
            plate_id = result.boxes.id[i].cpu().int().item() if result.boxes.is_track else None
            conf = result.boxes.conf[i].cpu().item()
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().int().numpy()

            # Put plate image into OCR queue
            plate_area = result.orig_img[y1:y2, x1:x2]
            self.ocr_queue.put((plate_id, plate_area))

            # Draw bounding box and OCR text
            annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (75, 25, 230), 5)
            annotated_frame = cv2.putText(
                annotated_frame, f'[{plate_id}]License Plate {conf:0.2f}', (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 25, 230), 2, cv2.LINE_AA
            )
            plate_text = self.ocr_dict[plate_id][0] if plate_id in self.ocr_dict else ''
            if plate_text:
                annotated_frame = cv2.putText(
                    annotated_frame, f'OCR: [{plate_text}]', (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 25, 230), 2
                )

            # frame_id, bndbox, plate_confidence, timestamp
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

            current_conf = self.detections_dict.get(plate_id, (None, None, -1))[2]
            if conf >= current_conf or current_conf == -1:
                self.detections_dict[plate_id] = (self.detection_counter, (x1, y1, x2, y2), conf, formatted_datetime)
            self.detection_counter += 1

        return annotated_frame

    def setup(self, path):
        self.ocr_queue = LifoQueue()
        self.ocr_dict = {}
        self.detections_dict = {}
        self.detection_counter = 0
        self.results = self.model.track(source=path, stream=True, verbose=False, device=self.device, imgsz=640,
                                        tracker=self.tracker, conf=self.conf_threshold)
