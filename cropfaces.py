import cv2
import os
import sys
import glob

def facecrop(image):
  facedata = "haarcascade_frontalface_default.xml"
  cascade = cv2.CascadeClassifier(cv2.data.haarcascades + facedata)

  img = cv2.imread(image)

  minisize = (img.shape[1],img.shape[0])
  miniframe = cv2.resize(img, minisize)

  faces = cascade.detectMultiScale(miniframe)
  counter = 0
  for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

    sub_face = img[y:y+h, x:x+w]
    fname, ext = os.path.splitext(image)
    cv2.imwrite(fname+"_cropped_"+str(counter)+ext, sub_face)
    counter += 1
  return

def process(root_dir):
  for path, _, files in os.walk('.'):
      for file in files:
        if file.endswith('.jpeg'):
          print(file)
          facecrop(os.path.join(path, file))

process("photos/")

