import numpy as np
import cv2
from numba import jit, prange

class ScaleError(Exception):
    pass

def get_edge_image(image):
    sobel_x = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    return cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY), ddepth=-1, kernel=sobel_x)

@jit(nopython=True, parallel=True)
def get_horizontal_line_image(image, line_size):
    # if len(image.shape) == 3 or not (
    #         np.array_equal(np.unique(image), [0]) or
    #         np.array_equal(np.unique(image), [255]) or
    #         np.array_equal(np.unique(image), [0, 255])
    # ):
    #     raise ScaleError("Image must be in binary format")

    result = np.zeros_like(image)
    rows, cols = image.shape

    for col in prange(cols):
        white_indices = np.where(image[:, col] == 255)[0]
        if white_indices.size == 0:
            continue
        diffs = np.diff(white_indices)
        breaks = np.where(diffs != 1)[0] + 1
        groups = np.split(white_indices, breaks)
        for group in groups:
            if group.size >= line_size:
                result[group[0]:group[-1]+1, col] = 255

    return result

@jit(nopython=True, parallel=True)
def get_vertical_line_image(image, line_size):
    # if len(image.shape) == 3 or not (
    #         np.array_equal(np.unique(image), [0]) or
    #         np.array_equal(np.unique(image), [255]) or
    #         np.array_equal(np.unique(image), [0, 255])
    # ):
    #     raise ScaleError("Image must be in binary format")

    result = np.zeros_like(image)
    rows, cols = image.shape

    for row in prange(rows):
        white_indices = np.where(image[row, :] == 255)[0]
        if white_indices.size == 0:
            continue
        diffs = np.diff(white_indices)
        breaks = np.where(diffs != 1)[0] + 1
        groups = np.split(white_indices, breaks)
        for group in groups:
            if group.size >= line_size:
                result[row, group[0]:group[-1]+1] = 255

    return result

def get_line_image(image, line_size):
    # if len(image.shape) == 3 or not (
    #         np.array_equal(np.unique(image), [0]) or
    #         np.array_equal(np.unique(image), [255]) or
    #         np.array_equal(np.unique(image), [0, 255])
    # ):
    #     raise ScaleError("Image must be in binary format")

    horizontal_line = get_horizontal_line_image(image, line_size)
    vertical_line = get_vertical_line_image(image, line_size)

    return np.bitwise_or(horizontal_line, vertical_line)

def get_edge_line_image(image, line_size):
    edge_image = get_edge_image(image)
    _, edge_image = cv2.threshold(edge_image, int(np.mean(edge_image)), 255, cv2.THRESH_BINARY)

    return get_line_image(edge_image, line_size)


if __name__ == '__main__':
    image = cv2.imread("C:/Users/alex6/Downloads/chess_board.png")
    image = cv2.resize(image, (640, 480))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("scaleImage", get_edge_line_image(image, 20))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

