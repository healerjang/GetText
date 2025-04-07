import cv2
import numpy as np
from . import image_util

def set_gaussian_noise(image, mean, sigma):
    gaussian_noise = np.random.normal(mean,sigma,image.shape)
    result = image.astype(np.float32) + gaussian_noise
    return np.clip(result,0,255).astype(np.uint8)

def set_salt_and_pepper_noise(image, noise_prob):
    random_matrix = np.random.rand(image.shape[0], image.shape[1])
    salt_threshold = noise_prob / 2
    pepper_threshold = 1 - (noise_prob / 2)
    image[random_matrix < salt_threshold] = 255
    image[random_matrix > pepper_threshold] = 0

    return image

def set_line_noise(image, line_length: tuple, line_thickness: tuple, num_lines: tuple, line_type='perpendicular'):
    image_shape = image.shape
    h,w = image_shape[0], image_shape[1]
    if len(image_shape) == 3:
        color_type = 'RGB'
    else:
        if np.array_equal(np.unique(image), [0, 255]):
            color_type = 'BINARY'
        else:
            color_type = 'GRAY'

    def __horizontal(x1, y1):
        x2, y2 = x1 + np.random.randint(*line_length), y1
        x2 = x2 if x2 <= w else w
        __set_line(x1, y1, x2, y2)

    def __vertical(x1, y1):
        x2, y2 = x1, y1 + np.random.randint(*line_length)
        y2 = y2 if y2 <= h else h
        __set_line(x1, y1, x2, y2)

    def __down_diag(x1, y1):
        length = np.random.randint(*line_length)
        x2, y2 = x1+length, y1+length
        x2 = x2 if x2 <= w else w
        y2 = y2 if y2 <= h else h
        __set_line(x1, y1, x2, y2)

    def __up_diag(x1, y1):
        length = np.random.randint(*line_length)
        x2, y2 = x1+length, y1-length
        x2 = x2 if x2 <= w else w
        y2 = y2 if y2 >= 0 else 0
        __set_line(x1, y1, x2, y2)

    def __random(x1, y1):
        x1, x2 = np.random.randint(0, w), np.random.randint(0, w)
        y1, y2 = np.random.randint(0, h), np.random.randint(0, h)
        __set_line(x1, y1, x2, y2)

    def __set_line(x1, y1, x2, y2):
        thickness = np.random.randint(*line_thickness)
        cv2.line(image, (x1, y1), (x2, y2), image_util.get_random_color(color_type), thickness)

    line_type_dic = {
        'horizontal': [__horizontal],
        'vertical': [__vertical],
        'down_diag': [__down_diag],
        'up_diag': [__up_diag],
        'random': [__random],
        'perpendicular': [__horizontal, __vertical],
        'all': [__horizontal, __vertical, __down_diag, __up_diag, __random],
        'all_less_random': [__horizontal, __vertical, __down_diag, __up_diag],
        'diagonal': [__down_diag, __up_diag]
    }

    if line_type not in line_type_dic:
        raise ValueError('Invalid line type')

    line_types = line_type_dic[line_type]

    for _ in range(np.random.randint(*num_lines)):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        np.random.choice(line_types)(x1, y1)

    return image


if __name__ == '__main__':
    image = cv2.imread("C:/Users/alex6/Downloads/pieces.jpg", 0)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('image', set_line_noise(image, (40, 80), (1, 5), (10, 20), line_type='all_less_random'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()