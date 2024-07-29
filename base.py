import cv2
import numpy as np
import face_recognition


# Load gambar dan merubah dari BGR ke RGB
gambarMegan = face_recognition.load_image_file('Images/Megan Fox.jpg')
gambarMegan = cv2.cvtColor(gambarMegan, cv2.COLOR_BGR2RGB)
# gambarTest = face_recognition.load_image_file('Images/Scarlett Johansson.jpg')
gambarTest = face_recognition.load_image_file('Images/Megan Fox Test.jpg') # menggunakan sample gambar dengan wajah dari sudut pandang lain
gambarTest = cv2.cvtColor(gambarTest, cv2.COLOR_BGR2RGB)


# Mendeteksi wajah dari gambar dengan menunjukkan kotak
lokasiWajah = face_recognition.face_locations(gambarMegan)[0]
encodeMegan = face_recognition.face_encodings(gambarMegan)[0]
cv2.rectangle(gambarMegan, (lokasiWajah[3], lokasiWajah[0]), (lokasiWajah[1], lokasiWajah[2]), (255, 0, 255), 2)

lokasiWajahTest = face_recognition.face_locations(gambarTest)[0]
encodeTest = face_recognition.face_encodings(gambarTest)[0]
cv2.rectangle(gambarTest, (lokasiWajahTest[3], lokasiWajahTest[0]), (lokasiWajahTest[1], lokasiWajahTest[2]), (255, 0, 255), 2)


# Menghitung jarak wajah, jika jarak < 0.5, berarti wajah cocok
hasil = face_recognition.compare_faces([encodeMegan], encodeTest)
jarakWajah = face_recognition.face_distance([encodeMegan], encodeTest)
print(hasil, jarakWajah)
cv2.putText(gambarTest, f'{jarakWajah[0]:.2f}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) # menampilkan text jarak antara wajah


cv2.imshow('Megan Fox', gambarMegan)
cv2.imshow('Test', gambarTest)
cv2.waitKey(0)