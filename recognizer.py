import cv2
from datetime import datetime
import pymysql


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = ["", "hero"]  # ID 1 = Muskan

cam = cv2.VideoCapture(0)

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        if confidence < 70:
            name = names[id]
            today = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now()

            db = pymysql.connect(
                host="localhost",
                user="root",
                password="password",      
                database="college_info"
            )

            cursor = db.cursor()

            # Check if today's attendance already exists
            cursor.execute(
             "SELECT * FROM attendence WHERE id=%s AND date=%s",
                (id, today)
            )

            record = cursor.fetchone()
            if record:
                in_time = record[2]

                if record[3] is None:        # out_time is NULL

                    minutes = (datetime.now() - in_time).total_seconds()/60

                if minutes >= 1:
                    cursor.execute(
                    "UPDATE attendence SET out_time=%s WHERE id=%s AND date=%s",
                        (datetime.now(), id, today)
                )
                    db.commit()

                    print("Exit Time Stored")

            # If no attendance exists for today, insert it
            if record is None:
                cursor.execute(
                    "INSERT INTO attendence (id, date,in_time) VALUES (%s, %s, %s)",
                    (id, today, current_time)
                )
                db.commit()
                print("entry time stored")

            cursor.close()
            db.close()

            

        else:
            name = "Unknown"

        cv2.putText(
            img,
            str(name),
            (x+5, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(10) & 0xff == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()