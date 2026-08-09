import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def add_person():
    person_id = input("Enter Person ID: ").strip()

    if not person_id.isdigit() or int(person_id) <= 0:
        print("Invalid ID.")
        return

    person_id = int(person_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("\nCamera started.")
    print("Look at the camera.")
    print("Press SPACE to cancel.")

    count = 0

    while count < 50:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            face = gray[y:y + h, x:x + w]

            count += 1

            filename = f"user.{person_id}.{count}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(filepath, face)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Images: {count}/50",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            break

        cv2.imshow("Add Person", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nCaptured {count} face images.")

    if count > 0:
        print(f"Images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    add_person()