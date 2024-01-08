from flask import Flask,request
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
app = Flask(__name__)


@app.route("/process",methods=['POST','GET'])
def hello():
    path = 'ImagesAttendance'
    images = []
    classNames = []
    mylist = os.listdir(path)
    print(mylist)
    for cls in mylist:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])
    print(classNames)

    def findEncoding(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(name):
        with open('Templates/Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                Date = now.strftime('%d.%m.%y')

                f.writelines(f'\n{name}, {dtString}, {Date}')
                print(f'\n{name},{dtString}')

    encodelisten = findEncoding(images)
    print("Encoding Completed")

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgs)
        encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodelisten, encodeFace)
            faceDis = face_recognition.face_distance(encodelisten, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 - 6, y2 - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        key=cv2.waitKey(1)
        if(key==ord('q')):
            print("Turning off camera.")
            cap.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            return "Attendance Marked Successfully"
            break

@app.route("/capture",methods=['POST','GET'])
def hello2():
    if(request.method == "POST"):
        name = request.form.get("fname")
    key = cv2.waitKey(1)
    webcam = cv2.VideoCapture(0)
    path = 'C:/Users/THENNARASU R/PycharmProjects/Face_Recognition/ImagesAttendance'
    while True:
        try:
            check, frame = webcam.read()
            print(check)  # prints true as long as the webcam is running
            print(frame)  # prints matrix values of each framecd
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            count = 0
            if key == ord('s'):
                count += 1
                cv2.imwrite(os.path.join(path, name+'.jpg'), img=frame)
                webcam.release()
                img_new = cv2.imread(os.path.join(path, name+'.jpg'), cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread(os.path.join(path, name+'.jpg'), cv2.IMREAD_ANYCOLOR)
                return "Image Saved Successfully"

                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()

                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    app.run()