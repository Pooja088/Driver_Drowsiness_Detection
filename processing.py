import cv2
import requests
import os
import time
from tensorflow import keras
import numpy as np

yawn_model = keras.models.load_model("./yawn_model.h5")
eye_model = keras.models.load_model("./eye_model.h5")

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

def get_cropped_image(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        (x, y, w, h) = faces[0]
        crop_face = gray[y:y+h,x:x+w]
        crop_face = cv2.resize(crop_face, (100, 100))
        crop_face = np.array(crop_face)
        crop_face = crop_face.astype("float32")
        crop_face /= 255
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        (x1, y1, w1, h1) = eyes[0]
        (x2, y2, w2, h2) = eyes[1]
        crop_eyes = [gray[y1:y1+h1,x1:x1+w1], gray[y2:y2+h2,x2:x2+w2]]
        crop_eyes_l = cv2.resize(crop_eyes[0], (100, 100))
        crop_eyes_l = np.array(crop_eyes_l)
        crop_eyes_l = crop_eyes_l.astype("float32")
        crop_eyes_l /= 255
        crop_eyes_r = cv2.resize(crop_eyes[1], (100, 100))
        crop_eyes_r = np.array(crop_eyes_r)
        crop_eyes_r = crop_eyes_r.astype("float32")
        crop_eyes_r /= 255
        return [crop_face.reshape((-1, 100, 100, 1)), crop_eyes_l.reshape((-1, 100, 100, 1)), crop_eyes_r.reshape((-1, 100, 100, 1))]
    except Exception as e:
        return None

def find_drowsy(url):
    file_id = url.split("/")[-2]
    destination = './input.mp4'
    download_file_from_google_drive(file_id, destination)

    cap = cv2.VideoCapture(destination)

    if not cap.isOpened():
        print("Error opening video file")

    results = []

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("frame", frame)
            frame_imgs = get_cropped_image(frame)
            if frame_imgs is not None:
                end_time = time.time()
                if end_time - start_time < 5:
                    continue
                yawn_prediction = yawn_model.predict(frame_imgs[0],batch_size = 1)
                eye1_prediction = eye_model.predict(frame_imgs[1],batch_size = 1)
                eye2_prediction = eye_model.predict(frame_imgs[2],batch_size = 1)
                if yawn_prediction[0][0] >= 0.5 or eye1_prediction[0][0] >= 0.5 and eye2_prediction[0][0] >= 0.5:
                    results.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                start_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    os.remove(destination)
    return results