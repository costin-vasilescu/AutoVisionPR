import numpy as np
import os
from PIL import Image, ExifTags


# Creates a list of weights with the weights provided in weights_dict
# and the remaining weights equally distributed among the remaining items
def create_weights_list(items, weights_dict):
    total_weight = sum(weights_dict.values())
    remaining_weight = 1 - total_weight
    if remaining_weight < 0:
        raise ValueError("Total weight provided in the dictionary should not exceed 1.")
    remaining_items = len(items) - len(weights_dict)
    split_remainder = remaining_weight / remaining_items

    weights_list = []
    for item in items:
        if item in weights_dict.keys():
            weights_list.append(weights_dict[item])
        else:
            weights_list.append(split_remainder)

    return weights_list


def random_digits(n):
    return ''.join(str(np.random.randint(0, 10)) for _ in range(n))


def random_letters(n):
    # alphabet without the letter Q because it is not allowed in Romanian license plates
    alphabet = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'

    # O and I cannot be the first letter in Romanian license plates
    letters_idx = [alphabet.find('O')]
    while letters_idx[0] == alphabet.find('O') or letters_idx[0] == alphabet.find('I'):
        letters_idx = np.random.randint(len(alphabet), size=n)
    return ''.join(alphabet[letters_idx[i]] for i in range(n))


def get_all_files(path):
    # Returns a list of all files in the directory and its subdirectories
    files = []
    for dir_path, _, file_names in os.walk(path):
        for file in file_names:
            files.append(os.path.relpath(os.path.join(dir_path, file), path))
    return files


def bbox_to_yolo(bbox, img_width, img_height):
    xmin, ymin, xmax, ymax = bbox

    center_x = (xmin + xmax) / 2 / img_width
    center_y = (ymin + ymax) / 2 / img_height
    bbox_width = (xmax - xmin) / img_width
    bbox_height = (ymax - ymin) / img_height

    return center_x, center_y, bbox_width, bbox_height


def yolo_to_bbox(yolo_annotation, img_width, img_height):
    center_x, center_y, bbox_width, bbox_height = yolo_annotation

    xmin = int((center_x - bbox_width / 2) * img_width)
    ymin = int((center_y - bbox_height / 2) * img_height)
    xmax = int((center_x + bbox_width / 2) * img_width)
    ymax = int((center_y + bbox_height / 2) * img_height)

    return xmin, ymin, xmax, ymax


def flip_bbox(bbox, img_width):
    xmin, ymin, xmax, ymax = bbox
    return [img_width - xmax, ymin, img_width - xmin, ymax]


def flip_yolobbox(bbox):
    x_center, y_center, width, height = bbox
    flipped_bbox = (1 - x_center, y_center, width, height)
    return flipped_bbox


def get_random_file(directory):
    # Using os.scandir instead of os.listdir to avoid loading the 300k image files from the CCPD dataset
    files = os.scandir(directory)
    chosen = next(files)  # get first file
    for i, file in enumerate(files, start=2):  # start from second file
        if np.random.randint(i):  # probability: 1/i
            continue  # skip with probability: 1-1/i
        chosen = file  # pick with probability: 1/i
    return chosen.name


def random_indices(lst, n):
    if n > len(lst):
        raise ValueError("n is larger than the number of elements in the list.")
    return list(np.random.choice(len(lst), size=n, replace=False))


def extract_from_indices(lst, indices):
    return [lst[i] for i in indices]


def cleanup_labels(folder_path):
    if not os.path.exists(folder_path):
        print("The specified folder doesn't exist!")
        return

    images_path = os.path.join(folder_path, 'images')
    labels_path = os.path.join(folder_path, 'labels')

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print("'images' or 'labels' folders not found in the given directory!")
        return

    # Get the list of image files (without extension)
    images = {os.path.splitext(f)[0] for f in os.listdir(images_path)}

    # Loop through each .txt file in the 'labels' directory
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            # Get the file name without extension
            label_name = os.path.splitext(label_file)[0]

            # If the label doesn't have a corresponding image, delete it
            if label_name not in images:
                os.remove(os.path.join(labels_path, label_file))
                print(f"Deleted: {label_file}")

    print("Cleanup finished.")

def auto_orient(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()
        if exif is not None and orientation in exif:
            orientation_value = exif[orientation]

            if orientation_value == 2:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 3:
                image = image.rotate(180)
            elif orientation_value == 4:
                image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 5:
                image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 6:
                image = image.rotate(-90, expand=True)
            elif orientation_value == 7:
                image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # In case the EXIF data is not present or there's some issue with processing it.
        pass

    return image
