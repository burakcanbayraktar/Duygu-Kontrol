import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kameran yok")

while True:
    ret, frame = cap.read()

    result = DeepFace.analyze(frame, actions=['emotion'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = faceCascade.detectMultiScale(gray,1.1, 4)

    for (x, y, z, w) in faces:
        cv2.rectangle(frame, (x, y), (x+z, y+w), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, result['dominant_emotion'],
                (50, 50),
                font, 3, (0, 255, 0),
                2,
                cv2.LINE_4)
    cv2.imshow("Ifadeleri Gosterme ", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
