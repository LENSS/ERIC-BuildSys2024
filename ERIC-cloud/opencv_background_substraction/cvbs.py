
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('2021-08-15T14-00-00.mp4') # day rain
# cap = cv.VideoCapture('2021-08-15T10-00-01.mp4') # day no rain
# cap = cv.VideoCapture('2021-06-01T03-30-01.mp4') # night rain
# cap = cv.VideoCapture('2021-06-07T23-30-01.mp4') # night no rain

# cap = cv.VideoCapture('2021-05-24T11-00-02.mp4') # rain stops in 2 mins

# cap = cv.VideoCapture('doorbell-20210415-183901.mp4') # radu home no rain
# cap = cv.VideoCapture('doorbell-20210415-180901.mp4') # radu home heavy rain

# AAU dataset
# cap = cv.VideoCapture('20180501_045959_72F2-comb-cfr-cfr-left.mp4') # no rain
# cap = cv.VideoCapture('20180428_140126_7847-comb-cfr-cfr-left.mp4') # very light rain
# cap = cv.VideoCapture('20180428_160145_098C-comb-cfr-cfr-left.mp4') # light rain
# cap = cv.VideoCapture('20180430_150135_6A92-comb-cfr-cfr-left.mp4') # heavy rain

# fgbg_sy = cv.bgsegm.createBackgroundSubtractorMOG()
# fgbg_sn = cv.bgsegm.createBackgroundSubtractorMOG()

# fgbg_sy = cv.createBackgroundSubtractorMOG2(detectShadows=True)
# fgbg_sn = cv.createBackgroundSubtractorMOG2(detectShadows=False)

# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# fgbg_sy = cv.bgsegm.createBackgroundSubtractorGMG()

fgbg_sy = cv.createBackgroundSubtractorKNN(detectShadows=True)
fgbg_sn= cv.createBackgroundSubtractorKNN(detectShadows=False)

frame_id = 102600

while True:

    # cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()

    # ret, frame = cap.read()
    # if frame is None:
    #     break

    # fgmask = fgbg.apply(frame)
    fgmask1 = fgbg_sy.apply(frame)
    fgmask2 = fgbg_sn.apply(frame)
    
    # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

    # # concatanate image Horizontally
    # Hori = np.concatenate((img1, img2), axis=1)
      
    # # concatanate image Vertically
    # Verti = np.concatenate((img1, img2), axis=0)
      
    # cv2.imshow('HORIZONTAL', Hori)
    # cv2.imshow('VERTICAL', Verti)

    cv.imshow('Frame', frame)
    cv.imshow('FGMASK Frame Shadow', fgmask1)
    # cv.imshow('FGMASK Frame NoShadow', fgmask2)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

    frame_id += 1

cap.release()
cv.destroyAllWindows()