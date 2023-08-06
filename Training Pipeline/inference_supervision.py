import supervision as sv
import cv2

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)


def process_results(result):
    detections = sv.Detections.from_yolov8(result)
    labels = []

    for detection, _, conf, _, _ in detections:
        labels.append(f'License Plate {conf:0.2f}')

    annotated_frame = box_annotator.annotate(
        scene=result.orig_img,
        detections=detections,
        labels=labels
    )
    return annotated_frame


def image_inference(model, src, device, show=False, save=False):
    results = model.predict(source=src, device=device, save=save)
    img = process_results(results[0])
    if show:
        cv2.imshow('AutoVisionPR', img)
        cv2.waitKey(-1)


def stream_inference(model, src, device, show=False, save=False):
    frame_count = 0

    for result in model.predict(source=src, device=device, save=save, stream=True, verbose=False):
        frame = process_results(result)
        if show:
            cv2.imshow('AutoVisionPR', frame)
            cv2.imwrite(f'frames/frame{frame_count}.png', frame)
            frame_count+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
