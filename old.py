import cv2
import numpy as np

# Load the sample images and their names
known_face_images = []
known_face_names = []

# Load the sample images
for i in range(1, 6):
    image = cv2.imread(f"known_faces_{i}.jpg")
    known_face_images.append(image)
    known_face_names.append(f"Face {i}")

# Create a known face encoding for each sample image
known_face_encodings = []
for image in known_face_images:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_encoding = image_gray[y:y+h, x:x+w]
        known_face_encodings.append(face_encoding)

# Initialize the camera
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    _, frame = camera.read()

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through each face location and compare it with the known face encodings
    for (x, y, w, h) in faces:
        face_encoding = frame_gray[y:y+h, x:x+w]
        for known_face_encoding in known_face_encodings:
            if np.array_equal(face_encoding, known_face_encoding):
                print(f"Found {known_face_names[known_face_encodings.index(known_face_encoding)]}!")

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Write the name of the person on the face
        cv2.putText(frame, known_face_names[known_face_encodings.index(face_encoding)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()