from PIL import Image, ImageEnhance
import numpy as np
import cv2
import math


def add_salt_and_pepper_noise(img, prob=0.01):
    noisy_image = np.copy(img)
    salt_count = np.ceil(prob * img.size * 0.5)
    pepper_count = np.ceil(prob * img.size * 0.5)

    coords = [np.random.randint(0, i - 1, int(salt_count)) for i in img.shape[:2]]
    for channel in range(img.shape[2]):
        noisy_image[tuple(coords + [channel])] = 255

    coords = [np.random.randint(0, i - 1, int(pepper_count)) for i in img.shape[:2]]
    for channel in range(img.shape[2]):
        noisy_image[tuple(coords + [channel])] = 0

    return noisy_image


def add_compression_artifacts(img, intensity=20):
    noise = np.random.randint(-intensity, intensity, img.shape, dtype=np.int16)
    noisy_image = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def adjust_rotation(img, limit=10):
    angle = np.random.uniform(-limit, limit)

    # Calculate the dimensions of the rotated image
    width, height = img.size
    rad_angle = math.radians(angle)
    new_width = int(abs(width * math.cos(rad_angle)) + abs(height * math.sin(rad_angle)))
    new_height = int(abs(width * math.sin(rad_angle)) + abs(height * math.cos(rad_angle)))

    # Create a blank canvas
    canvas = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))

    # Rotate the image
    rotated_image = img.rotate(-angle, resample=Image.BICUBIC, expand=True)

    # Calculate the position to paste the rotated image onto the canvas
    left = (new_width - rotated_image.width) // 2
    top = (new_height - rotated_image.height) // 2

    canvas.paste(rotated_image, (left, top), rotated_image)

    return canvas


def adjust_brightness(img, brightness=None):
    brightness_factor = np.random.uniform(0.85, 1.15) if brightness is None else brightness
    enhancer = ImageEnhance.Brightness(img)
    adjusted_image = enhancer.enhance(brightness_factor)

    return adjusted_image


def blur_area(img, bbox):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    xmin, ymin, xmax, ymax = bbox

    region = np.copy(img[ymin:ymax, xmin:xmax])
    blurred_region = cv2.GaussianBlur(region, (15, 15), 0)
    img[ymin:ymax, xmin:xmax] = blurred_region

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def resize_to_bbox(img, bbox_width, bbox_height, scaling_factor=1.0):
    target_width = int(bbox_width * scaling_factor)
    target_height = int(bbox_height * scaling_factor)
    resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)

    return resized_img


def darken_white_pixels(img, threshold=250, reduce_by=50):
    pixels = img.load()

    for i in range(img.size[0]):  # for every col:
        for j in range(img.size[1]):  # for every row
            r, g, b, a = pixels[i, j]

            # check if the pixel is white or near white
            if r > threshold and g > threshold and b > threshold:
                # darken the pixel
                r, g, b = max(0, r - reduce_by), max(0, g - reduce_by), max(0, b - reduce_by)
                pixels[i, j] = (r, g, b, a)

    return img
