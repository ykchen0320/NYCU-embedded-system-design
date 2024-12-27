# OpenCV program to detect face in real time
# import libraries of python OpenCV 
# where its functionality resides
import cv2 
import time

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

i = 1

# loop runs if capturing has been initialized.
while 1: 
    # reads frames from a camera
    ret, img = cap.read() 

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        roi_gray = gray[y:y+h, x:x+w]

        face = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_AREA)

        print(f"saved: unknownfaces/{i}.png")

        cv2.imwrite(f"unknownfaces/{i}.png".format(i), face)
        i += 1

    time.sleep(1)

# Close the window
cap.release()
