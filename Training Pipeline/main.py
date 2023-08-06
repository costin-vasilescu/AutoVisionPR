from ultralytics import YOLO
from lanyocr import LanyOcr
from queue import LifoQueue
from utils import match_plate_text
import threading
import re
import numpy as np
import torch
import cv2
import time

plate_model = YOLO('./weights/my_dataset_v2.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ocr = LanyOcr(
    detector_name='easyocr_craft',
    recognizer_name='paddleocr_en_ppocr_v3',
    merger_name='lanyocr_craftbased',
    merge_boxes_inference=False,
    merge_rotated_boxes=True,
    merge_vertical_boxes=True,
    use_gpu=False if device == 'cpu' else True,
    disable_angle_classifier=True,
    debug=False
)

ocr_queue = LifoQueue()
ocr_dict = {}


class OCRThread(threading.Thread):
    def __init__(self, upscale=False):
        super().__init__()
        self._stop_event = threading.Event()
        self.upscale = upscale

    def run(self):
        while not self._stop_event.is_set():
            try:
                plate_id, plate_image = ocr_queue.get(timeout=5)
                # img = Image.fromarray(plate_image)
                # img.save(f'./ocr/{plate_id}.jpg')
            except Exception as e:
                print(e)
                continue

            if plate_id is None or plate_image.size == 0:
                continue
            ocr_result = ocr.infer(plate_image)

            best_score = ocr_dict.get(plate_id, (None, 0))[1]
            for result in ocr_result:
                text = re.sub(r"[^a-zA-Z0-9]", "", result.text)
                score = match_plate_text(text)
                plate_score = np.average([score, result.prob], weights=[0.7, 0.3])

                # print(f'text score: {score}, final score: {plate_score} for this text: {text}\n')
                if plate_score >= best_score:
                    ocr_dict[plate_id] = (text, plate_score)
                    best_score = plate_score

    def stop(self):
        self._stop_event.set()


def detect_plates(path):
    t = OCRThread()
    t.start()

    previous_time = time.perf_counter()
    results = plate_model.track(source=path, show=False, stream=True, verbose=False, device=device, imgsz=640, conf=0.5)
    for result in results:
        annotated_frame = result.orig_img.copy()
        for i in range(len(result)):
            pass
            plate_id = result.boxes.id[i].cpu().int().item() if result.boxes.is_track else None
            if plate_id is None:
                continue
            conf = result.boxes.conf[i].cpu().item()
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().int().numpy()

            plate_area = result.orig_img[y1:y2, x1:x2]
            ocr_queue.put((plate_id, plate_area))

            annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (75, 25, 230), 5)
            annotated_frame = cv2.putText(annotated_frame, f'[{plate_id}]License Plate {conf:0.2f}', (x1, y1 - 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (75, 25, 230), 2, cv2.LINE_AA)

            plate_text = ocr_dict[plate_id][0] if plate_id in ocr_dict else ''
            annotated_frame = cv2.putText(annotated_frame, f'OCR: [{plate_text}]', (x1, y2 + 30),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (75, 25, 230), 2)

        current_time = time.perf_counter()
        fps = 1 / (current_time - previous_time)
        annotated_frame = cv2.putText(
            annotated_frame, str(int(fps)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3
        )
        previous_time = current_time

        cv2.imshow('AutoVisionPR', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            t.stop()
            break

    # When video ends stop OCR thread
    t.stop()


def main():
    path = './testing_data/videos/IMG_0448.mov'
    # path = './testing_data/videos/cameratest.mkv'
    detect_plates(path)


if __name__ == '__main__':
    main()
