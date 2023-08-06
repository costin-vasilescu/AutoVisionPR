from ultralytics import YOLO
import torch
import os

model = YOLO('./weights/my_dataset_v2.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_annotation(bbox, file, detection_error):
    x, y, w, h = bbox
    x, y, w, h = x.item(), y.item(), w.item(), h.item()

    w *= 1 + detection_error
    h *= 1 + detection_error
    file.write(f'0 {x} {y} {w} {h}\n')


def annotate_plates(path, detection_error=0):
    if not os.path.exists('annotator_output'):
        os.mkdir('annotator_output')

    for file in os.listdir(path):
        image_path = os.path.join(path, file)
        results = model.predict(source=image_path, show=False, device=device, verbose=False)
        with open('./annotator_output/' + file.replace(os.path.splitext(file)[1], '.txt'), 'w') as f:
            print(f'{file} - {len(results[0].boxes)} detections')
            for box in results[0].boxes:
                save_annotation(box.xywhn[0], f, detection_error)


def main():
    path = r'./testing_data/images'
    annotate_plates(path)


if __name__ == '__main__':
    main()
