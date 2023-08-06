from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from utils import create_weights_list, random_digits, random_letters
import numpy as np
import os


class PlateType(Enum):
    STANDARD = 0
    PROVISIONAL = 1
    ELECTRIC = 2
    MOTORCYCLE = 3


class LicensePlate:
    plate_font_size = {
        'template1.png': 65,  # 81
        'template2.png': 50,
        'template3.png': 70  # 86
    }
    starting_text_position = {
        'template1.png': (50, 16),  # (54, 1)
        'template2.png': (45, 12),
        'template3.png': (50, 13)
    }
    plate_color = {
        PlateType.STANDARD: (0, 0, 0),
        PlateType.PROVISIONAL: (255, 0, 0),
        PlateType.ELECTRIC: (0, 128, 0),
        PlateType.MOTORCYCLE: (0, 0, 0)
    }
    plate_type_weights_dict = {
        PlateType.STANDARD: 0.92,
        PlateType.PROVISIONAL: 0.05,
        PlateType.MOTORCYCLE: 0.0
    }
    county_weights_dict = {
        'B': 0.2,
        'CJ': 0.1,
        'CT': 0.1
    }
    template_weights_dict = {
        'template1.png': 0.9,
        'template2.png': 0.0
    }

    with open('./resources/counties.txt', 'r') as f:
        counties = f.read().replace(' ', '').split(',')

    template_path = './resources/template/'
    font_path = './resources/font/DIN 1451 Std Mittelschrift.otf'

    county_weights = create_weights_list(counties, county_weights_dict)
    plate_type_weights = create_weights_list(list(PlateType), plate_type_weights_dict)
    template_weights = create_weights_list(os.listdir(template_path), template_weights_dict)

    def __init__(self, plate_type, plate_text=None, templates=None):
        self.plate_type = self.random_plate_type() if plate_type is None else plate_type
        self.plate_text = self.random_plate_text() if plate_text is None else plate_text
        self.templates = os.listdir(self.template_path) if templates is None else templates

    def random_plate_type(self):
        return np.random.choice(list(PlateType), p=self.plate_type_weights)

    def random_county_code(self):
        return np.random.choice(self.counties, p=self.county_weights)

    def random_plate_text(self):
        plate_text = [self.random_county_code()]
        random_number = np.random.rand()

        if self.plate_type == PlateType.PROVISIONAL:
            if random_number < 0.75 or plate_text[0] == 'B':
                plate_text.append('0' + random_digits(5))
            else:
                plate_text.append('0' + random_digits(4))
        else:
            # Standard, Electric, Motorcycle
            if plate_text[0] == 'B' and random_number < 0.62:
                plate_text.append(random_digits(3))
            else:
                plate_text.append(random_digits(2))
            if self.plate_type == PlateType.MOTORCYCLE:
                plate_text.append('\n')
            plate_text.append(random_letters(3))
        return plate_text

    def random_template(self):
        if self.plate_type == PlateType.PROVISIONAL or self.plate_type == PlateType.ELECTRIC:
            return self.template_path + 'template1.png'
        elif self.plate_type == PlateType.MOTORCYCLE:
            return self.template_path + 'template2.png'

        return self.template_path + np.random.choice(self.templates, p=self.template_weights)

    def generate_image(self):
        template = self.random_template()
        template_img = Image.open(template)
        font = ImageFont.truetype(self.font_path, self.plate_font_size[template.split('/')[-1]])
        text_position = self.starting_text_position[template.split('/')[-1]]
        draw = ImageDraw.Draw(template_img)

        # Calculate optimal spacing between plate text elements
        full_text = ''
        for text in self.plate_text:
            full_text += text
            if text == '\n':
                break
        text_spacing = int((template_img.width - text_position[0] - font.getbbox(full_text)[2]) / len(self.plate_text))

        for i, text in enumerate(self.plate_text):
            text_size = font.getbbox(text)
            text_size = (text_size[2], text_size[3])

            # Draw shadow to give depth effect
            # for offset in range(2):
            #     draw.text((text_position[0] - offset, text_position[1] - offset), text, fill='gray', font=font)

            if i == 0 and self.plate_type == PlateType.MOTORCYCLE:
                text_spacing = 10
                current_width = font.getbbox(full_text)[2] + text_spacing
                center_position = int((template_img.width - self.starting_text_position[template.split('/')[-1]][0] - current_width) / 2)
                center_position += self.starting_text_position[template.split('/')[-1]][0]
                text_position = (center_position, text_position[1])
            if text == '\n':
                current_width = font.getbbox(self.plate_text[-1])[2]
                center_position = int((template_img.width - current_width) / 2) + self.starting_text_position[template.split('/')[-1]][1]
                text_position = (center_position, text_position[1] + text_size[1] + 10)
                continue
            draw.text(text_position, text, font=font, fill=self.plate_color[self.plate_type])
            text_position = (text_position[0] + text_size[0] + text_spacing, text_position[1])

        return template_img
