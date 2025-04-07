import os
import random

import cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont, TTLibError
import numpy as np

from h5py_dataset import H5pyDatafile
from image_util import get_random_crp_image, over_draw_image
from video_to_data import VideoToData

# 모델의 train image와 label image를 생성하는 코드.
# draw_font_image_font 함수는 BGRA로 배경 이미지에 랜덤 폰트, 랜덤 단어, 랜덤 포지션에 텍스트 이미지를 그린 이미지를 train,
# train 이미지와 동일한 크기의 검은색 이미지에 train 이미지에 덧씌운 텍스트 이미지를 같은 위치에 덧씌운 이미지를 label로 생성.
# 두 이미지 다 BGRA의 128*128 이미지.

# draw_font_image_position 함수는 BGR 배경 이미지에 draw_font_image_font와 동일한 방법으로 train image를 생성.
# label은 word를 글자로 쪼개서 글자의 중심 position에 흰색 픽셀을 검은색 이미지에 추가.
# train은 BGR 128*128 이미지, label은 GRAY 128*128 이진 이미지.

# 해당 코드들을 동영상의 모든 프레임을 대상으로 추가하여 데이터셋을 생성.


def __get_word__(item):
    global word_list
    word = item.decode("utf-8")
    word_list.append(word)


word_dataset = H5pyDatafile("E:/SpliceImageTextData/dataset/words.h5")
word_list = []
word_dataset.get_all('scaling_word', __get_word__)


def get_random_font_path():
    font_dir_path = "E:/SpliceImageTextData/fonts"
    font_files = os.listdir(font_dir_path)
    random_font_file = np.random.choice(font_files)
    return os.path.join(font_dir_path, random_font_file)


def is_font(font_path, word):
    try:
        font = TTFont(font_path)
    except TTLibError:
        return False
    cmap = {}
    for table in font['cmap'].tables:
        cmap.update(table.cmap)
    for char in word:
        if ord(char) not in cmap:
            return False
    return True

def get_char_size(word, font):
    char_list = list(word)
    result = []
    for char in char_list:
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        result.append((width, height))

    return result

def draw_font_image_position(word, image, font_size=15, font_color=(0, 0, 0)):
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    black_image = np.zeros((h, w))

    while True:
        font_path = get_random_font_path()
        if is_font(font_path, word):
            break

    font = ImageFont.truetype(font_path, size=font_size)
    bbox = font.getbbox(word)
    font_width, font_height = int(bbox[2] - bbox[0]) + 1, int(bbox[3] - bbox[1]) + 1

    start_x, start_y = random.randint(0, w - font_width), random.randint(0, h - font_height)
    draw = ImageDraw.Draw(pil_image)
    draw.text((start_x, start_y), word, font=font, fill=font_color)

    char_sizes = get_char_size(word, font)
    for char_size in char_sizes:
        x = start_x + (char_size[0] // 2)
        y = start_y + font_height - (char_size[1] // 2)
        black_image[y:y + 1, x:x + 1] = 255
        start_x += char_size[0]

    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return image, black_image


def draw_font_image_font(word, image, font_size=15, font_color=(0, 0, 0)):
    color_avg = sum(font_color) / len(font_color)
    blank_color = 0 if color_avg < 127 else 255

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    black_image = np.full((h, w, 3), blank_color, dtype=np.uint8)
    black_pil_image = Image.fromarray(black_image)

    while True:
        font_path = get_random_font_path()
        if is_font(font_path, word):
            break

    font = ImageFont.truetype(font_path, size=font_size)
    bbox = font.getbbox(word)
    font_width = int(bbox[2] - bbox[0]) + 1
    font_height = int(bbox[3] - bbox[1]) + 1
    font_width = font_width if font_width == 0 else font_width - 1
    font_height = font_height if font_height == 0 else font_height - 1

    start_x = random.randint(0, w - font_width)
    start_y = random.randint(0, h - font_height)

    draw = ImageDraw.Draw(pil_image)
    draw.text((start_x, start_y), word, font=font, fill=font_color)

    draw_black = ImageDraw.Draw(black_pil_image)
    draw_black.text((start_x, start_y), word, font=font, fill=font_color)

    image_bgra = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGRA)
    black_bgr = cv2.cvtColor(np.array(black_pil_image), cv2.COLOR_RGB2BGR)

    mask_image = np.zeros((h, w, 4), dtype=np.uint8)
    mask_image[..., :3] = black_bgr

    blank_mask = np.all(black_bgr == blank_color, axis=-1)
    mask_image[..., 3] = np.where(blank_mask, 0, 255)
    mask_image[blank_mask, :3] = 0

    return image_bgra, mask_image


class Maker(VideoToData):
    def __init__(self, video_path, dataset_object):
        self.dataset_object = dataset_object
        super().__init__(video_path, skip=9)

    def __start__(self, image):
        crp_image, train_image, word_label_base_image, word = get_image(image)
        print(crp_image.shape)
        print(train_image.shape)
        print(word_label_base_image.shape)
        print(word)
        pass

    def __to_data__(self, image):
        try:
            crp_image, train_image, word_label_base_image, word = get_image(image)
        except cv2.error:
            train_image = np.zeros((128, 128, 4), dtype=np.uint8)
            word_label_base_image = np.zeros((128, 128, 4), dtype=np.uint8)
            crp_image = np.zeros((128, 128, 3), dtype=np.uint8)
            word = ""

        # cv2.imshow("crp_image", crp_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(word)
        # cv2.imshow("train_image", train_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow("word_label_base_image", word_label_base_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        self.dataset_object.save('korean_image', train_image)
        self.dataset_object.save('korean_label', word_label_base_image)
        self.dataset_object.save('korean_base', crp_image)
        self.dataset_object.save('korean_word', word)

        if random.random() < 0.2:
            self.dataset_object.save('korean_image', cv2.cvtColor(crp_image, cv2.COLOR_BGR2BGRA))
            self.dataset_object.save('korean_base', crp_image)
            self.dataset_object.save('korean_label', np.zeros((128, 128, 4), dtype=np.uint8))
            self.dataset_object.save('korean_word', "")


    def __end__(self, image):
        pass


def get_crp_image(image, size=128):
    h, w = image.shape[:2]
    min_shape = min(h, w)
    max_size = min_shape - (min_shape % size)
    max_magnification = max_size / size
    magnification = random.randint(1, max_magnification)
    scaling_size = size * magnification
    crp_image = get_random_crp_image(image, (scaling_size, scaling_size), (h - h * 0.08, h, 0, w * 0.27))
    return cv2.resize(crp_image, (size, size), interpolation=cv2.INTER_AREA)


def get_image(image):
    crp_image = get_crp_image(image)
    word = random.choice(word_list)

    font_size = random.randint(12, 18)
    train_image, label_image = draw_font_image_font(word, crp_image, font_size, font_color=tuple(np.random.randint(0, 256, size=3)))
    return crp_image, train_image, label_image, word


def main():
    image_dataset_path = "E:/SpliceImageTextData/dataset/word_image_train_label2.h5"
    if os.path.exists(image_dataset_path):
        os.remove(image_dataset_path)

    image_dataset = H5pyDatafile(image_dataset_path)
    maker = Maker("E:/SpliceImageTextData/video/cute_animals.mp4", image_dataset)
    maker.__run__()
    print(image_dataset.shape())

def view_dataset():
    image_dataset = H5pyDatafile("E:/SpliceImageTextData/dataset/word_image_train_label2.h5")
    for i in range(58330):
        train = image_dataset.get('korean_image', i)
        label = image_dataset.get('korean_label', i)
        label[label == 1] = 255

        cv2.imshow("train", train)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("label", label)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # view_dataset()