import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

cascade = cv2.CascadeClassifier(CASCADE_PATH)

if cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Camera started.")
print("Look at the camera.")
print("Press SPACE to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()