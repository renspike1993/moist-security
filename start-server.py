from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import mysql.connector
import faiss
import os
from collections import Counter

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


matching_threshold = 0.4

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
    print("⚠️ Warning: No face data found in database.")
    full_names = []
    index = None  # No index if there's no face data
else:
    face_encodings = [np.frombuffer(r[1], dtype=np.float64).astype('float32') for r in results]
    full_names = [r[2] for r in results]
    
    index = faiss.IndexFlatL2(len(face_encodings[0]))
    index.add(np.array(face_encodings))

    matching_threshold = 0.4  # Adjust as needed

def apply_augmentations(image):
    """Apply multiple augmentations to an image and return a list of augmented images."""
    augmented_images = [image]

    # Flip image horizontally
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotate image
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    augmented_images.append(rotated)

    # Adjust brightness
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Increase brightness
    augmented_images.append(bright)

    return augmented_images
@app.route('/upload', methods=['POST'])
def upload_and_recognize():
    try:
        if 'face' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('face')
        if len(files) != 3:
            return jsonify({'error': 'Exactly 3 images are required'}), 400

        recognized_ids = []

        for file in files:
            if file.filename == '':
                return jsonify({'error': 'One or more files are empty'}), 400

            image = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'error': 'Invalid image file'}), 400

            # Apply augmentations
            augmented_images = apply_augmentations(frame)

            for aug_frame in augmented_images:
                rgb_frame = cv2.cvtColor(aug_frame, cv2.COLOR_BGR2RGB)
                face_encodings_live = face_recognition.face_encodings(rgb_frame)

                if not face_encodings_live:
                    continue  # Skip if no face is detected

                # Recognize the face
                query = np.array([face_encodings_live[0]]).astype('float32')
                D, I = index.search(query, 1)
                recognized_id = full_names[I[0][0]] if D[0][0] < matching_threshold else "Unknown"
                recognized_ids.append(recognized_id)

        if not recognized_ids:
            return jsonify({'error': 'No valid faces detected after augmentation'}), 400

        # Get most common recognized ID
        most_common_id, count = Counter(recognized_ids).most_common(1)[0]

        if count > 2:
            return jsonify({
                'message': 'Recognition completed',
                'recognized_id': most_common_id,
                'occurrences': count
            }), 200            
        else:
            return jsonify({
                'message': 'Recognition completed',
                'recognized_id':'No face detected',
                'occurrences': 0
            }), 200            
                    
    finally:
        print("Error")
if __name__ == '__main__':
    try:
        print("🚀 Starting Flask server on port 6600...")
        app.run(host='0.0.0.0', port=6600, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")
