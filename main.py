import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# Mendeteksi file yang ada di dalam folder Images
lokasi = 'Images' # lokasi gambar
gambar = []
namaClass = [] # list yang akan digunakan untuk menyimpan file gambar tanpa ekstensi
myList = os.listdir(lokasi)
# print(myList)
for i in myList:
    currentImg = cv2.imread(f'{lokasi}/{i}')
    gambar.append(currentImg)
    namaClass.append(os.path.splitext(i)[0])
print('Nama-nama yang terdeteksi : ', namaClass)


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


# Fungsi absensi
def absensi(nama):
    with open('Kehadiran.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if nama not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{nama},{dtString}')


# Fungsi untuk menutup kamera
def tutupKamera():
    kamera.release()
    cv2.destroyAllWindows()
    exit()


# Pencarian wajah
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1092)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
# Perulangan untuk mendeteksi wajah
while True:
    success, img = kamera.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        cocok = face_recognition.compare_faces(listEncoding, encodeFace)
        jarakWajah = face_recognition.face_distance(listEncoding, encodeFace)
        # print(jarakWajah)
        matchIndex = np.argmin(jarakWajah)

        if cocok[matchIndex]:
            nama = namaClass[matchIndex].upper()
            print('Wajah terdeteksi : ', nama)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'Nama : '+nama, (x1+6, y1-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            absensi(nama)

    cv2.imshow('Detektor Wajah', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tutupKamera()