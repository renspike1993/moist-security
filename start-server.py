from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import mysql.connector
import faiss
import os

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MySQL connection setup
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="moist_logs"
    )
    cursor = db.cursor()
    cursor.execute("""
        SELECT person_faces.file_name, person_faces.encoding, person.fullname
        FROM person_faces
        JOIN person ON person_faces.owner_id = person.id
    """)
    results = cursor.fetchall()
    cursor.close()
    db.close()
except Exception as e:
    print(f"Database error: {e}")
    results = []

if not results:
    print("No face data found in database.")
    exit()

face_encodings = [np.frombuffer(r[1], dtype=np.float64).astype('float32') for r in results]
full_names = [r[2] for r in results]
index = faiss.IndexFlatL2(len(face_encodings[0]))
index.add(np.array(face_encodings))
matching_threshold = 0.4

@app.route('/upload', methods=['POST'])
def upload_and_recognize():
    if 'face' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('face')  # Get all uploaded files
    if len(files) != 5:
        return jsonify({'error': 'Exactly 5 images are required'}), 400

    recognized_ids = []

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'One or more files are empty'}), 400

        image = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image file'}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings_live = face_recognition.face_encodings(rgb_frame)
        if not face_encodings_live:
            return jsonify({'error': 'No face detected in one or more images'}), 400

        # Recognize the face in the image
        query = np.array([face_encodings_live[0]]).astype('float32')
        D, I = index.search(query, 1)
        recognized_id = full_names[I[0][0]] if D[0][0] < matching_threshold else "Unknown"
        recognized_ids.append(recognized_id)

    # Check if all recognized IDs are the same
    if all(id == recognized_ids[0] for id in recognized_ids):
        return jsonify({'message': 'All images have the same ID', 'recognized_id': recognized_ids[0]}), 200
    else:
        return jsonify({'error': 'Images do not have the same ID', 'recognized_ids': recognized_ids}), 400

if __name__ == '__main__':
    try:
        print("Starting Flask server on port 6600...")
        app.run(host='0.0.0.0', port=6600, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")