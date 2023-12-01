import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time

detector = HandDetector(maxHands=2)
cap = cv2.VideoCapture(0)
offset = 100
imgsize = 500

folder = "DATA/TIGER"
counter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

            imgcrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = imgsize / h
                w_cal = math.ceil(k * w)
                img_resize = cv2.resize(imgcrop, (w_cal, imgsize))
                img_resized_shape = img_resize.shape
                w_gap = math.ceil((imgsize - w_cal) / 2)

                imgwhite[:, w_gap : w_gap + w_cal] = img_resize

            else:
                k = imgsize / w
                h_cal = math.ceil(k * h)
                img_resize = cv2.resize(imgcrop, (imgsize, h_cal))
                img_resized_shape = img_resize.shape
                h_gap = math.ceil((imgsize - h_cal) / 2)

                imgwhite[h_gap : h_gap + h_cal, :] = img_resize

            cv2.imshow('Imagecrop', imgcrop)
            cv2.imshow('Whiteimage', imgwhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite)
        print(counter)
