import cv2
import numpy as np
import face_recognition

gambarMegan = face_recognition.load_image_file('Images/Megan Fox.jpg')
gambarMegan = cv2.cvtColor(gambarMegan, cv2.COLOR_BGR2RGB)
gambarTest = face_recognition.load_image_file('Images/Megan Fox Test.jpg')
gambarTest = cv2.cvtColor(gambarTest, cv2.COLOR_BGR2RGB)

cv2.imshow('Megan Fox', gambarMegan)
cv2.imshow('Test', gambarTest)
cv2.waitKey(0)