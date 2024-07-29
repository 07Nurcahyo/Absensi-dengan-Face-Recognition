import cv2
import numpy as np
import face_recognition
import os


# Mendeteksi file yang ada di dalam folder Images
lokasi = 'Images' # lokasi gambar
gambar = []
nama = [] # list yang akan digunakan untuk menyimpan file gambar tanpa ekstensi
myList = os.listdir(lokasi)
# print(myList)
for i in myList:
    currentImg = cv2.imread(f'{lokasi}/{i}')
    gambar.append(currentImg)
    nama.append(os.path.splitext(i)[0])
print('Nama-nama yang terdeteksi : ', nama)


# Fungsi pencarian encoding
def cariEncoding(gambar):
    encodeList = []
    for img in gambar:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


listEncoding = cariEncoding(gambar)
print('Jumlah : ', len(listEncoding))
print('Proses Encoding Selesai!')