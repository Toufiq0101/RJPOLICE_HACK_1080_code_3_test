import cv2
import os
import time
from datetime import datetime

def save(img, name, bbox, width=180, height=227):
    x, y, w, h = bbox
    imgCrop = img[y:h, x:w]
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name + ".jpg", imgCrop)

def fetch_face_data(video_path):
    today_date = datetime.today().date()
    formatted_date = today_date.strftime("%d-%m-%y")
    if not os.path.exists(
        f"D:/ProjectX/data_storage/{formatted_date}/facial_data"
    ) and not os.path.isdir(f"D:/ProjectX/data_storage/{formatted_date}/facial_data"):
        os.makedirs(f"D:/ProjectX/data_storage/{formatted_date}/facial_data")
    new_path = f"D:/ProjectX/data_storage/{formatted_date}/facial_data"
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    file_names = []
    while True:
        current_timestamp = int(time.time())
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for counter, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 255, 220), 1)
            save(gray, os.path.join(new_path, str(current_timestamp)), (x, y, x + w, y + h))
            file_names.append(f"{os.path.join(new_path, str(current_timestamp))}")
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(file_names)


if __name__ == "__main__":
    video_path = "D:/ProjectX/face/fc.mp4"
    fetch_face_data(video_path)