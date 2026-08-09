import os
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TRAINED_DIR = os.path.join(BASE_DIR, "trained")
MODEL_PATH = os.path.join(TRAINED_DIR, "training.yml")

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


def train_model():

    print("\n========================================")
    print("           TRAINING MODEL")
    print("========================================")

    if not os.path.exists(OUTPUT_DIR):
        print("Output folder does not exist.")
        return

    image_paths = [
        os.path.join(OUTPUT_DIR, filename)
        for filename in os.listdir(OUTPUT_DIR)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        print("No face images found.")
        print("Add a person first.")
        return

    faces = []
    ids = []

    for image_path in image_paths:

        try:
            filename = os.path.basename(image_path)

            parts = filename.split(".")

            if len(parts) < 3 or parts[0].lower() != "user":
                print(f"Skipping invalid file: {filename}")
                continue

            person_id = int(parts[1])

            image = Image.open(image_path).convert("L")
            face = np.array(image, dtype=np.uint8)

            faces.append(face)
            ids.append(person_id)

        except Exception as error:
            print(f"Skipping {image_path}: {error}")

    if not faces:
        print("No valid training images found.")
        return

    print(f"\nTraining with {len(faces)} images...")
    print(f"People IDs: {sorted(set(ids))}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        faces,
        np.array(ids)
    )

    os.makedirs(TRAINED_DIR, exist_ok=True)

    recognizer.save(MODEL_PATH)

    print("\nTraining completed successfully.")
    print(f"Model saved at:")
    print(MODEL_PATH)



def recognize_person():

    print("\n========================================")
    print("          FACE RECOGNITION")
    print("========================================")

    if not os.path.exists(MODEL_PATH):
        print("Trained model not found.")
        print("Please train the model first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        recognizer.read(MODEL_PATH)
    except Exception as error:
        print(f"Could not load trained model: {error}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("\nCamera started.")
    print("Look at the camera.")
    print("Press SPACE to exit.")

    while True:

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

            person_id, confidence = recognizer.predict(face)

            # LBPH confidence is actually a distance.
            # Lower value means a better match.
            if confidence < 70:
                name = f"Person {person_id}"
                label = f"{name} ({confidence:.1f})"
            else:
                label = f"Unknown ({confidence:.1f})"

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    print("\n========================================")
    print("       FACE RECOGNITION SYSTEM")
    print("========================================")

    print("\n1. Add Person")
    print("2. Train Model")
    print("3. Recognize Person")

    choice = input("\nEnter your choice: ").strip()

    if choice == "1":
        add_person()

    elif choice == "2":
        train_model()

    elif choice == "3":
        recognize_person()

    else:
        print("Invalid choice.")  