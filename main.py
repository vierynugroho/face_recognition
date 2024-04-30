import cv2

# Membuat objek CascadeClassifier untuk deteksi wajah
face_ref = cv2.CascadeClassifier("face_ref.xml")
# Mengaktifkan kamera webcam
camera = cv2.VideoCapture(0)


def face_detection(frame):
    # Konversi frame ke skala grayscale untuk deteksi wajah
    optimize_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Deteksi wajah di dalam frame
    faces = face_ref.detectMultiScale(optimize_frame, scaleFactor=1.1, minNeighbors=5)
    return faces


def drawer_box(frame, faces, name):
    for (x, y, w, h) in faces:
        # Membuat kotak sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        # Menulis nama di atas wajah
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def close_window():
    # Menghentikan penggunaan kamera
    camera.release()
    # Menutup semua jendela OpenCV
    cv2.destroyAllWindows()
    # Keluar dari program
    exit()

def main():
    name = "Viery"
    while True:
        # Membaca frame dari kamera
        _, frame = camera.read()
        # Deteksi wajah di dalam frame
        faces = face_detection(frame)
        # Menggambar kotak sekitar wajah dan menulis nama
        drawer_box(frame, faces, name)
        # Menampilkan frame dengan label "Deteksi Wajah"
        cv2.imshow("Deteksi Wajah", frame)

        # Jika tombol 'q' ditekan, maka program akan berhenti
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()


if __name__ == '__main__':
    main()