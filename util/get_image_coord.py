import cv2
import numpy as np

def main(image):
    image_filter = image[20:34, 28:46]
    filter_image = cv2.filter2D(image, -1, image_filter).astype(np.uint64)
    threshold = 0.9 * np.sum(image_filter.astype(np.float32) ** 2)
    print(threshold)
    print(np.unique(filter_image))
    filter_image[filter_image < threshold] = 0
    print(np.unique(filter_image))
    cv2.imshow('edge', filter_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = cv2.imread("C:/Users/alex6/Pictures/Screenshots/edge.png", 0)
    main(image)