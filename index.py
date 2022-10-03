import numpy as np
import cv2

vid = cv2.VideoCapture("20221002_144334.mp4")

frames = []
fps = vid.get(5)
frameWidth = 500
frameHeight = 500

centers = []

while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:
        cv2.imshow("Frame: ", frame)

        scale = 50
        w = int(frame.shape[1] * scale / 100)
        h = int(frame.shape[0] * scale / 100)
        dim = (w, h)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        frameWidth = w
        frameHeight = h

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([23, 75, 86], dtype="uint8")
        upper = np.array([62, 254, 255], dtype="uint8")

        mask = cv2.inRange(img, lower, upper)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        c= max(contours, key = cv2.contourArea)

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cX, cY), 1, (255, 0, 0), -1)

        frames.append(frame)
        centers.append((cX, cY))

        key = cv2.waitKey(20)
    else:
        break;

vid.release()
cv2.destroyAllWindows()

out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (frameWidth, frameHeight))

for i in range(len(frames)):
    out.write(frames[i])
out.release()

f = open("out.txt", "w")
for i in range(len(centers)):
    f.write(str(centers[i][0]) + ", " + str(centers[i][1]) + "\n")

f.close()
