from PIL import Image
from LP import LicensePlate
from utils import get_all_files, bbox_to_yolo, yolo_to_bbox, flip_bbox, flip_yolobbox, get_random_file, random_indices, \
    extract_from_indices, auto_orient
from image_filters import blur_area, resize_to_bbox, darken_white_pixels, add_compression_artifacts, \
    add_salt_and_pepper_noise, adjust_brightness
import numpy as np
import albumentations as A
import os


class LicensePlateDataset:
    background_path = './resources/background/'
    output_path = './output/'
    required_dirs = [output_path + 'images/', output_path + 'labels/']

    file_name = 'PlateDataset{}.png'
    image_number = 0
    plate_counter = 0

    plate_transform = A.Compose([
        A.RandomGamma(gamma_limit=(80, 100), p=0.5),
        A.GaussNoise(p=0.35),
        # A.CoarseDropout(max_holes=8, max_height=3, max_width=3, fill_value=0, p=0.35),
        A.MotionBlur(blur_limit=3, p=0.35),
        A.Perspective(scale=(0.02, 0.05), fit_output=True, p=0.8),
    ])

    def __init__(self, backgrounds=None, max_plates=5, template=None):
        self.backgrounds = get_all_files(self.background_path) if backgrounds is None else backgrounds
        self.max_plates = max_plates
        self.template = template

    def random_background(self):
        return self.background_path + np.random.choice(self.backgrounds)

    def check_intersection(self, current_plates_data, new_plate_data):
        new_plate_position = new_plate_data[0]
        new_plate_size = new_plate_data[1]
        x1, y1, w1, h1 = new_plate_position[0], new_plate_position[1], new_plate_size[0], new_plate_size[1]

        intersection = False
        for plate_position, plate_size in current_plates_data:
            x2, y2, w2, h2 = plate_position[0], plate_position[1], plate_size[0], plate_size[1]
            if not (x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2):
                intersection = True

        return intersection

    def random_plate_position(self, plate_size, img_size):
        x = np.random.randint(0, img_size[0] - plate_size[0])
        y = np.random.randint(0, img_size[1] - plate_size[1])
        return x, y

    def random_plate(self, plate_type):
        new_plate = LicensePlate(plate_type)
        plate_img = np.array(darken_white_pixels(new_plate.generate_image(), reduce_by=25))

        rnd = np.random.rand()
        if rnd < 0.4:
            plate_img = add_salt_and_pepper_noise(plate_img)
        plate_img = self.plate_transform(image=plate_img)['image']
        rnd = np.random.rand()
        if rnd < 0.35:
            plate_img = add_compression_artifacts(plate_img)

        plate_img = Image.fromarray(plate_img)
        plate_img = adjust_brightness(plate_img)

        return plate_img

    def generate_yolo_labels(self, plate_data, img_size, img_name, verbose):
        label_file = img_name.replace('.png', '.txt')

        with open(self.output_path + 'labels/' + label_file, 'w') as f:
            for plate_position, plate_size in plate_data:
                x_center = (plate_position[0] + plate_size[0] / 2) / img_size[0]
                y_center = (plate_position[1] + plate_size[1] / 2) / img_size[1]
                width = plate_size[0] / img_size[0]
                height = plate_size[1] / img_size[1]
                f.write(f'0 {x_center} {y_center} {width} {height}\n')

        if verbose:
            print(f'Generated label {label_file}\n')

    def generate(self, n, plate_type=None, verbose=True):
        # Generate required directories
        [os.makedirs(d) for d in self.required_dirs if not os.path.exists(d)]

        for _ in range(n):
            plate_data = []
            img = Image.open(self.random_background())

            # Resize image if too small to generate a variety of resolutions
            if img.width < 1000 or img.height < 1000:
                w = np.random.randint(1000, 2000)
                h = np.random.randint(1000, 2000)
                img = img.resize((w, h))

            plate_count = np.random.randint(1, self.max_plates + 1)
            for _ in range(plate_count):
                plate_img = self.random_plate(plate_type)

                # Generate random position for plate until it doesn't intersect with other plates
                new_plate_position = self.random_plate_position(plate_img.size, img.size)
                while self.check_intersection(plate_data, (new_plate_position, plate_img.size)):
                    new_plate_position = self.random_plate_position(plate_img.size, img.size)
                plate_data.append((new_plate_position, plate_img.size))

                img.paste(plate_img, new_plate_position, plate_img.convert('RGBA'))

            img_file = self.file_name.format(self.image_number)
            img.save(self.output_path + 'images/' + img_file)
            if verbose:
                print(f'Generated image {img_file} with {plate_count} plates: {plate_data}')
            self.generate_yolo_labels(plate_data, img.size, img_file, verbose)
            self.image_number += 1

    def generate_from_preannotated(self, path, n=None, per_image=2, plate_type=None, random_sampling=False,
                                   verbose=True):
        # Generate required directories
        [os.makedirs(d) for d in self.required_dirs if not os.path.exists(d)]

        image_files = os.listdir(path + '/images')
        label_files = os.listdir(path + '/labels')
        if len(image_files) != len(label_files):
            raise Exception('Number of images and labels does not match')

        # Randomly sample n images from the dataset
        if random_sampling:
            if n is None:
                raise Exception('n must be specified when random_sampling is True')
            indices = random_indices(image_files, n)
            image_files = extract_from_indices(image_files, indices)
            label_files = extract_from_indices(label_files, indices)

        # Read image and annotation data
        for image, label in zip(image_files, label_files):
            for i in range(per_image):
                img = Image.open(path + '/images/' + image)
                img = auto_orient(img)

                with open(path + '/labels/' + label, 'r') as f:
                    # Flip the second half of the images
                    flip = False if i < per_image / 2 else True
                    if flip:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    yolo_label = []
                    for line in f.readlines():
                        # Yolo annotation format: class x_center y_center box_width box_height
                        yolo_bbox = list(map(float, line.split(' ')[1:]))
                        if flip:
                            yolo_bbox = flip_yolobbox(yolo_bbox)

                        # Convert yolo annotation to xmin, ymin, xmax, ymax
                        bbox = list(yolo_to_bbox(yolo_bbox, img.width, img.height))

                        # Blur original license plate
                        img = blur_area(img, bbox)

                        # Resize plate image to match bbox
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                        plate_img = self.random_plate(plate_type)
                        plate_img = resize_to_bbox(plate_img, bbox_width, bbox_height, scaling_factor=1.08)

                        # Paste plate image at the same location as label
                        img.paste(plate_img, (bbox[0], bbox[1]), plate_img.convert('RGBA'))

                        # Adjust annotation to account for image filters
                        bbox[2] = bbox[0] + plate_img.width
                        bbox[3] = bbox[1] + plate_img.height
                        yolo_bbox = bbox_to_yolo(bbox, img.width, img.height)
                        yolo_label.append(yolo_bbox)

                # Save image and label file
                extension = os.path.splitext(image)[1]
                img_file = image.replace(extension, '') + '_' + self.file_name.format(self.image_number)
                img.save(self.output_path + 'images/' + img_file)
                label_file = img_file.replace('.png', '.txt')
                with open(self.output_path + 'labels/' + label_file, 'w') as f:
                    for yolo_bbox in yolo_label:
                        f.write(f'0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n')

                if verbose:
                    print(f'Generated image {img_file} with {len(yolo_label)} plates')
                    print(f'Generated label {label_file}\n')
                self.image_number += 1
                self.plate_counter += len(yolo_label)

            if n is not None:
                if self.image_number == n * per_image:
                    break

        print(f'Generated {self.image_number} images with {self.plate_counter} license plates')

    def generate_from_ccpd(self, path, n=None, per_image=2, plate_type=None, verbose=True):
        # Generate required directories
        [os.makedirs(d) for d in self.required_dirs if not os.path.exists(d)]

        for _ in range(n):
            for i in range(per_image):
                # Sample random image from CCPD dataset
                img_name = get_random_file(path)
                img = Image.open(os.path.join(path, img_name))

                # Get bounding box from image name, CCPD annotation format: area-tiltdegree-x1&y1_x2&y2-...
                x1y1, x2y2 = img_name.split('-')[2].split('_')
                bbox = list(map(int, x1y1.split('&'))) + list(map(int, x2y2.split('&')))

                # Flip the second half of the images
                flip = False if i < per_image / 2 else True
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    bbox = flip_bbox(bbox, img.width)

                # Blur original license plate
                img = blur_area(img, bbox)

                # Resize plate image to match bbox
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                plate_img = self.random_plate(plate_type)
                plate_img = resize_to_bbox(plate_img, bbox_width, bbox_height, scaling_factor=1.08)

                # Paste plate image at the same location as label
                img.paste(plate_img, (bbox[0], bbox[1]), plate_img.convert('RGBA'))

                # Adjust annotation to account for image filters
                bbox[2] = bbox[0] + plate_img.width
                bbox[3] = bbox[1] + plate_img.height
                yolo_bbox = bbox_to_yolo(bbox, img.width, img.height)

                # Save image and label file
                extension = os.path.splitext(img_name)[1]
                img_file = f'ccpdimg{self.image_number}' + extension
                img.save(self.output_path + 'images/' + img_file)
                label_file = img_file.replace(extension, '.txt')
                with open(self.output_path + 'labels/' + label_file, 'w') as f:
                    f.write(f'0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}')

                if verbose:
                    print(f'Generated image {img_file}')
                    print(f'Generated label {label_file}\n')
                self.image_number += 1
                self.plate_counter += 1

        print(f'Generated {self.image_number} images with {self.plate_counter} license plates from CCPD dataset')
