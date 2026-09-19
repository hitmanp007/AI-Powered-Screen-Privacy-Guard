import cv2
import os

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Error: Haar Cascade file could not be loaded.")
    print(cv2.data.haarcascades)
    exit()

face_id = input("Enter User ID: ")

count = 0

os.makedirs("dataset", exist_ok=True)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        count += 1

        cv2.imwrite(
            f"dataset/User.{face_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        cv2.imshow("Capture Face", img)

    k = cv2.waitKey(100) & 0xff 

    if k == ord("q"):
        break
    elif count >= 50:
        break

cam.release()
cv2.destroyAllWindows()