from queue import Queue
import threading
import numpy as np
import cv2
import supervision as sv


class InfAlt:
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    def __init__(self, model, plate_model, device):
        self.model = model
        self.plate_model = plate_model
        self.device = device
        self.vehicle_queue = Queue()

    def detect_vehicles(self, src):
        for result in self.model.predict(source=src, device=self.device, stream=True, classes=[2, 5, 7], verbose=False):
            vehicle_detections = sv.Detections.from_yolov8(result)
            self.vehicle_queue.put((result.orig_img, vehicle_detections.xyxy))

    def process_results(self, frame, vehicle_detections):
        for detection in vehicle_detections:
            x1, y1, x2, y2 = detection.astype('int')
            results = self.plate_model.predict(source=frame[y1:y2, x1:x2], verbose=False, device=self.device)

            plate_detections = sv.Detections.from_yolov8(results[0])
            if not plate_detections.empty():
                x1, y1 = detection[:2]
                plate_detections.xyxy += np.float32([x1, y1, x1, y1])
                labels = []
                for _, _, plate_conf, _, _ in plate_detections:
                    labels.append(f'License Plate {plate_conf:0.2f}')
                frame = self.box_annotator.annotate(
                    scene=frame,
                    detections=plate_detections,
                    labels=labels
                )
        return frame

    def image_inference(self, src, show=False, save=False):
        results = self.model.predict(source=src, device=self.device, classes=[2, 5, 7])
        vehicle_detections = sv.Detections.from_yolov8(results[0])
        annotated_frame = self.process_results(results[0].orig_img, vehicle_detections.xyxy)

        if show:
            cv2.imshow('AutoVisionPR', annotated_frame)
            cv2.waitKey(-1)
        if save:
            cv2.imwrite('inference_alternative.png', annotated_frame)

    def stream_inference(self, src, show=False, save=False):
        t = threading.Thread(target=self.detect_vehicles, args=(src,))
        t.start()

        while True:
            try:
                frame, vehicle_detections = self.vehicle_queue.get(timeout=10)
                annotated_frame = self.process_results(frame, vehicle_detections)
            except:
                print('Video stream ended')
                break

            if show:
                cv2.imshow('AutoVisionPR', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
