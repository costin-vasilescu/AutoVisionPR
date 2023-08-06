from ultralytics import YOLO
from inference_alternative import InfAlt
import inference as inf
import inference_supervision as inf_sv
import torch
import os


def main():
    model = YOLO('./weights/my_dataset_v1.pt')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path = r'D:\AutoVisionPR Data\Testing'
    images_path = os.path.join(path, 'Images')
    videos_path = os.path.join(path, 'Videos')
    images = os.listdir(images_path)
    videos = os.listdir(videos_path)

    print('Images:')
    for idx, image in enumerate(images):
        print(f'[{idx}] {image}')
    print('\nVideos:')
    for idx, video in enumerate(videos):
        print(f'[{idx}] {video}')

    src1 = os.path.join(images_path, images[3])
    src2 = os.path.join(videos_path, videos[1])

    # inf.image_inference(model, src1, device, show=True, save=False)
    # inf.stream_inference(model, src2, device, show=True, save=False)

    # inf_sv.image_inference(model, src1, device, show=True, save=True)
    inf_sv.stream_inference(model, src2, device, show=True, save=False)

    # inf_alt = InfAlt(YOLO('./weights/yolov5su.pt'), model, device)
    # #inf_alt.image_inference(src1, show=True, save=False)
    # inf_alt.stream_inference(src2, show=True, save=False)


if __name__ == '__main__':
    main()
