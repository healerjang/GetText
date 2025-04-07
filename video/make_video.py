import mss
import keyboard
import os
import numpy as np
import cv2
import time

def make_video(save_path, video_name, height, width, fps=20.0):

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        print("Press 1 to start recording")

        while True:
            is_pressed_f6 = False
            if keyboard.is_pressed('1'):
                print("Start recording")
                print("Press 2 to end recording")
                video_count = sum(1 for entry in os.scandir(save_path) if entry.is_file())

                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                img = cv2.resize(img, (width, height))

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(f'{save_path}/{video_name + video_count}.avi', fourcc, fps,
                                      (width, height), isColor=False)
                out.write(img)

                while True:
                    sct_img = sct.grab(monitor)
                    img = np.array(sct_img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    img = cv2.resize(img, (width, height))
                    out.write(img)

                    if keyboard.is_pressed('2'):
                        print("End recording")
                        is_pressed_f6 = True
                        break

                    time.sleep(1.0 / fps)

                out.release()

            if is_pressed_f6:
                break