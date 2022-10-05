import numpy as np
import cv2
import glob

# termination criteria
critera = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare obj points
objp = np.zeros((14*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:14, 0:10].T.reshape(-1, 2)

# Arrays to store obj points & img points
objPoints = [] # 3d point in real world space
imgPoints = [] # 2d points in image plane

images = glob.glob("calibrationPhotos/*.jpg")

for fname in images:
    img = cv2.imread(fname)

    scale = 20
    w = int(img.shape[1] * scale / 100)
    h = int(img.shape[0] * scale / 100)
    dim = (w, h)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (13, 9), None)

    if ret == True:
        objPoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), critera)
        imgPoints.append(corners)

        cv2.drawChessboardCorners(img, (13, 9), corners2, ret)
        cv2.imshow("", img)
        cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

img = cv2.imread("test.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imshow("", dst)
