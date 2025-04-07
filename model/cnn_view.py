import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

DATA_PATH = "E:/SpliceImageTextData/dataset/word_image_train_label2.h5"
MODEL_PATH = "D:/SpliceImageTextData\model/cnn_text_white_pixel_hard.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def nearest_multiple_of_four(n):
    return int(round(n / 4)) * 4

test_img2 = cv2.imread("C:/Users/alex6/Downloads/chess_board2.png")
test_img = test_img2.copy()
test_img = cv2.resize(test_img, (0, 0), fx=2, fy=2)
h, w = test_img.shape[:2]
black_image = np.zeros((h, w), np.uint8)
white_dot_filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])


while True:
    curr_h, curr_w = test_img.shape[:2]
    input_img = np.expand_dims(test_img, axis=0)
    pred_mask = model.predict(input_img)[0]
    pred_vis = (pred_mask * 255).astype(np.uint8).squeeze()
    _, pred_vis = cv2.threshold(pred_vis, 127, 255, cv2.THRESH_BINARY)
    pred_vis = cv2.resize(pred_vis, (w, h), interpolation=cv2.INTER_AREA)
    black_image[pred_vis == 255] = 255

    if curr_h < 256 or curr_w < 256:
        break

    test_img = cv2.resize(test_img, (nearest_multiple_of_four(curr_w * 0.67), nearest_multiple_of_four(curr_h * 0.67)), interpolation=cv2.INTER_AREA)


black_image = cv2.filter2D(black_image, -1, white_dot_filter)

# 시각화
plt.imshow(black_image, cmap='gray')
plt.title("Predicted Mask for Larger Image")
plt.axis('off')
plt.show()

cv2.imshow("Predicted Mask", test_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
