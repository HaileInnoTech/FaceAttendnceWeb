from flask import Flask, render_template, request, redirect, url_for, make_response, Response
import os
import cvzone
from threading import Thread
import shutil
import numpy as np
from datetime import datetime
import time
import base64
import pytz
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import pickle
import cv2
import mediapipe as mp
from google.cloud.storage import transfer_manager

current_date = datetime.now(pytz.timezone('US/Eastern')).strftime('%m/%d/%Y %H:%M:%S')
# Fetch the service account key JSON file contents
cred = credentials.Certificate('key.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'storageBucket': 'faceimagebucket-test',
    'databaseURL': 'https://fingerprintdb3-default-rtdb.firebaseio.com/'
})

app = Flask(__name__)
UPLOAD_FOLDER = 'Images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def generate_frames():
    # encodeGenerator()
    time.sleep(5)
    print("Loading Encode File ...")
    file = open('EncoderFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, peopleIds = encodeListKnownWithIds
    # print(peopleIds)
    print(" File Loaded")
    while video.isOpened():
        success, frame = video.read()
        if success:
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            faceCurFrame = face_recognition.face_locations(imgS)
            enCodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(enCodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(matches)
                print(faceDis)
                matchIndex = np.argmin(faceDis)
                if faceDis[matchIndex] < 0.5:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, bbox, rt=0)
                    id = peopleIds[matchIndex]
                    data = retrieveAttendence(id)
                    name = data['Full_Name']
                    last_CheckIn = data['Last_check']
                    updateAttendence(stuId=data['ID'], name=data['Full_Name'],
                                     email=data['Email'])  # bị update liên tục, cần chạy 1 lần
                    position = (x1, y1 - 10)  # Above the bounding box
                    last_checkin_position = (x1, y1 + 30)
                    cv2.putText(frame, name, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(frame, last_CheckIn, last_checkin_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                else:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, bbox, rt=0)
                    cv2.putText(frame, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    stuId = request.form['studentID']
    email = request.form['email']

    IDList = retrieveIDList()
    IDList = list(data.keys())
    print(data)
    number = 'sws00129'
    if stuId in IDList:
        return "ID already exists. Please choose a different one."
    else:
        imageData64 = request.form['capturedPhoto']
        uploadNewUser(stuId, name, email)
        photo_binary = base64.b64decode(imageData64.split(',')[1])
        uploadNewFace(photo_binary, stuId)
        time.sleep(5)
        return redirect(url_for('index'))
def uploadNewUser(stuId, name, email):
    ref = db.reference('FaceInformation')
    users_ref = ref.child(f'{stuId}')
    users_ref.set({
        'ID': f'{stuId}',
        'Image': f'{stuId}.png',
        'Full_Name': f'{name}',
        'Date Register': current_date,
        'Email': f'{email}',
    })
    ref = db.reference('FaceAttendnce')
    users_ref = ref.child(f'{stuId}')
    users_ref.set({
        'ID': f'{stuId}',
        'Image': f'{stuId}.png',
        'Full_Name': f'{name}',
        'Email': f'{email}',
    })
def uploadNewFace(photo_binary, stuId):
    bucket = storage.bucket()
    blob = bucket.blob(f'Images/{stuId}.png')
    blob.upload_from_string(photo_binary, content_type='image/png')
def encodeGenerator():
    myfir = "Images"
    shutil.rmtree(myfir)
    time.sleep(5)
    # if file.endswith('.png'):
    #     os.remove(file)
    #     print('Delete file ok')
    bucket = storage.bucket()
    blob_names = [blob.name for blob in bucket.list_blobs(max_results=50)]
    results = transfer_manager.download_many_to_path(bucket, blob_names, max_workers=8)
    for name, result in zip(blob_names, results):

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, f'Images' + name))

    folderPath = "Images"
    pathList = os.listdir(folderPath)
    imageList = []
    peopleIds = []

    for path in pathList:
        imageList.append(cv2.imread(os.path.join(folderPath, path)))
        peopleIds.append(os.path.splitext(path)[0])

    def FindEncodings(imageList):
        encodeList = []
        for img in imageList:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    print("Encoding Started ....")
    encodeListKnown = FindEncodings(imageList)
    encodeListKnownWithIds = [encodeListKnown, peopleIds]
    print("Encoding Completed ")

    file = open("EncoderFile.p", 'wb')
    pickle.dump(encodeListKnownWithIds, file)
    file.close()
    print("File Saved")
def updateAttendence(stuId, name, email):
    current = current_date
    data = retrieveAttendence(stuId)
    counter = int(data['Times'])
    last = data['Attendent_time']
    print(f'Last Check:{last}')

    print(f'Current Check:{current}')
    ref = db.reference('FaceAttendence')
    users_ref = ref.child(f'{stuId}')
    users_ref.update({
        'ID': f'{stuId}',
        'Image': f'{stuId}.png',
        'Full_Name': f'{name}',
        'Attendent_time': f'{current}',
        'Last_check': f'{last}',
        'Email': f'{email}',
    })
    last = datetime.strptime(last, '%m/%d/%Y %H:%M:%S')
    current = datetime.strptime(current, '%m/%d/%Y %H:%M:%S')
    if last.date() == current.date():
        print('Already check In')
    else:
        counter += 1
        users_ref.update({
            'Times': str(counter)
        })
        print('Check In Succesfully')
def retrieveAttendence(stuId):
    ref = db.reference(f'FaceAttendence/{stuId}')
    return ref.get()


def retrieveIDList():
    ref = db.reference(f'FaceInformation/')
    data = ref.order_by_key().get()
    return data


@app.route('/sync_data', methods=['get'])
def sync_data():
    print("Processing")
    encodeGenerator()
    generate_frames()
    response = make_response('')
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(debug=True)
