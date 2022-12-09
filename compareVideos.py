import cv2
import numpy as np

capture = cv2.VideoCapture("/Users/niloofar/Documents/Projects/Video_OF/videoData/outputVideo_moviF_pretrained.mp4")
capture2 = cv2.VideoCapture("/Users/niloofar/Documents/Projects/Video_OF/videoData/normal.mp4")
counter = 0
while True:
    f, frame = capture.read()
    f2, frame2 = capture2.read()
    if(not f2 or not f):
        break
    # frame = cv2.GaussianBlur(frame,(15,15),0)
    # frame2 = cv2.GaussianBlur(frame2, (15, 15), 0)
    counter = 0

    res = np.sum(np.abs(frame - frame2))/(512*512)
    counter += res
print(counter)