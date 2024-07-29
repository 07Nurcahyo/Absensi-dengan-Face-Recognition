import cv2
import numpy as np
import face_recognition

gambarMegan = face_recognition.load_image_file('Images/Megan Fox.jpg')
gambarMegan = cv2.cvtColor(gambarMegan, cv2.COLOR_BGR2RGB)
gambarTest = face_recognition.load_image_file('Images/Scarlett Johansson.jpg')
gambarTest = cv2.cvtColor(gambarTest, cv2.COLOR_BGR2RGB)

lokasiWajah = face_recognition.face_locations(gambarMegan)[0]
encodeMegan = face_recognition.face_encodings(gambarMegan)[0]
cv2.rectangle(gambarMegan, (lokasiWajah[3], lokasiWajah[0]), (lokasiWajah[1], lokasiWajah[2]), (255, 0, 255), 2)

lokasiWajahTest = face_recognition.face_locations(gambarTest)[0]
encodeTest = face_recognition.face_encodings(gambarTest)[0]
cv2.rectangle(gambarTest, (lokasiWajahTest[3], lokasiWajahTest[0]), (lokasiWajahTest[1], lokasiWajahTest[2]), (255, 0, 255), 2)

cv2.imshow('Megan Fox', gambarMegan)
cv2.imshow('Test', gambarTest)
cv2.waitKey(0)