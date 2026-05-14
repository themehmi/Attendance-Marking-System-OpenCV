import sys
import os
import io

# Fix Windows console encoding for DeepFace emoji output
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import traceback
import re
import time
from flask import Flask, render_template, Response, request, jsonify
import base64
import cv2
import face_recognition
import numpy as np
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from deepface import DeepFace

app = Flask(__name__)

# SETUP DATABASE
DB_FILE = "attendance.db"

def init_db():
    """Creates the database and table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            emotion TEXT DEFAULT '',
            age TEXT DEFAULT '',
            gender TEXT DEFAULT '',
            race TEXT DEFAULT ''
        )
    ''')
    # Migration: add new columns if they don't exist (for existing DBs)
    try:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN emotion TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN age TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN gender TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE attendance_logs ADD COLUMN race TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

# Initialize the database immediately
init_db()


# LOAD DATA ON STARTUP
DATASET_DIR = Path("dataset_extracted")
DATASET_DIR.mkdir(exist_ok=True)

known_encodings = []
known_names = []

print("Loading dataset and encoding faces. Please wait...")
for person_name in os.listdir(DATASET_DIR):
    person_path = DATASET_DIR / person_name
    if not person_path.is_dir():
        continue

    for image_name in os.listdir(person_path):
        image_path = person_path / image_name
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"Loaded {len(known_encodings)} faces. Starting app...")


# DEEPFACE ANALYSIS CACHE
# Stores { name: { "emotion": str, "age": int, "gender": str, "race": str, "timestamp": float } }
analysis_cache = {}
CACHE_TTL = 10  # seconds before re-analyzing a known person
UNKNOWN_CACHE_TTL = 5  # seconds for unknown faces


def get_cached_analysis(name):
    """Return cached analysis if still fresh, else None."""
    if name in analysis_cache:
        entry = analysis_cache[name]
        ttl = UNKNOWN_CACHE_TTL if name == "Unknown" else CACHE_TTL
        if time.time() - entry["timestamp"] < ttl:
            return entry
    return None


def run_deepface_analysis(face_img):
    """
    Run DeepFace.analyze on a cropped face image.
    Returns dict with emotion, age, gender, race or None on failure.
    """
    try:
        results = DeepFace.analyze(
            face_img,
            actions=['emotion', 'age', 'gender', 'race'],
            enforce_detection=False,
            silent=True
        )
        # DeepFace returns a list; take the first result
        result = results[0] if isinstance(results, list) else results

        return {
            "emotion": result.get("dominant_emotion", "N/A"),
            "age": int(result.get("age", 0)),
            "gender": result.get("dominant_gender", "N/A"),
            "race": result.get("dominant_race", "N/A"),
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"DeepFace analysis error: {e}")
        return None


# LOGGING LOGIC
marked_names = set()

def mark_attendance(name, analysis_data=None):
    if name not in marked_names:
        marked_names.add(name)
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        emotion = ""
        age = ""
        gender = ""
        race = ""
        if analysis_data:
            emotion = analysis_data.get("emotion", "")
            age = str(analysis_data.get("age", ""))
            gender = analysis_data.get("gender", "")
            race = analysis_data.get("race", "")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO attendance_logs (name, date, time, emotion, age, gender, race)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, current_date, current_time, emotion, age, gender, race))
        conn.commit()
        conn.close()

        print(f"Attendance Logged in Database for: {name} "
              f"[emotion={emotion}, age={age}, gender={gender}, race={race}]")


# EMOJI MAPPINGS
EMOTION_EMOJIS = {
    "happy": "😊", "sad": "😢", "angry": "😠", "surprise": "😲",
    "fear": "😨", "disgust": "🤢", "neutral": "😐"
}


# WEBCAM FRAME PROCESSING (Cloud-compatible)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']

        header, encoded = data.split(",", 1)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        faces_analysis = []  # analysis data to send to frontend

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

            # Scale face location back to full resolution
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Crop face from full frame for DeepFace analysis
            # Add padding for better analysis accuracy
            h, w = frame.shape[:2]
            pad = 30
            crop_top = max(0, top - pad)
            crop_bottom = min(h, bottom + pad)
            crop_left = max(0, left - pad)
            crop_right = min(w, right + pad)
            face_crop = frame[crop_top:crop_bottom, crop_left:crop_right]

            # Check cache or run analysis
            cache_key = name if name != "Unknown" else f"Unknown_{left}_{top}"
            analysis = get_cached_analysis(cache_key)

            if analysis is None and face_crop.size > 0:
                analysis = run_deepface_analysis(face_crop)
                if analysis:
                    analysis_cache[cache_key] = analysis

            # Mark attendance with analysis data
            if name != "Unknown":
                mark_attendance(name, analysis)

            # ── Draw on frame ──

            # Face rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 140, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Semi-transparent label background
            label_text = name
            if analysis:
                emoji_char = EMOTION_EMOJIS.get(analysis.get("emotion", "").lower(), "")
                label_text = f"{name}"

            # Name label (above box)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (left, top - th - 14), (left + tw + 10, top), color, -1)
            cv2.putText(frame, label_text, (left + 5, top - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Analysis labels (below box)
            if analysis:
                info_lines = [
                    f"Emotion: {analysis.get('emotion', 'N/A')}",
                    f"Age: ~{analysis.get('age', 'N/A')}",
                    f"Gender: {analysis.get('gender', 'N/A')}",
                    f"Race: {analysis.get('race', 'N/A')}",
                ]
                # Draw dark background panel below face box
                line_h = 22
                panel_h = len(info_lines) * line_h + 10
                panel_w = 220
                panel_x = left
                panel_y = bottom + 5

                # Clamp panel to frame bounds
                if panel_y + panel_h > h:
                    panel_y = top - panel_h - th - 20
                if panel_x + panel_w > w:
                    panel_x = w - panel_w - 5

                # Draw semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y),
                              (panel_x + panel_w, panel_y + panel_h),
                              (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

                # Draw text lines
                for i, line in enumerate(info_lines):
                    y_pos = panel_y + 18 + i * line_h
                    cv2.putText(frame, line, (panel_x + 8, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)

            # Build frontend data
            face_data = {
                "name": name,
                "box": {"top": top, "right": right, "bottom": bottom, "left": left}
            }
            if analysis:
                face_data["analysis"] = {
                    "emotion": analysis.get("emotion", "N/A"),
                    "age": analysis.get("age", "N/A"),
                    "gender": analysis.get("gender", "N/A"),
                    "race": analysis.get("race", "N/A"),
                }
            faces_analysis.append(face_data)

        _, buffer = cv2.imencode('.jpg', frame)
        out_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': f"data:image/jpeg;base64,{out_b64}",
            'faces': faces_analysis
        })

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500



#  NEW: USER REGISTRATION ROUTE
@app.route('/register', methods=['POST'])
def register():
    """
    Accepts a name and an array of base64-encoded face images (minimum 5).
    Saves all valid images to dataset_extracted/<name>/ and hot-reloads
    every face encoding into memory — no server restart required.
    """
    MIN_PHOTOS = 5
    MAX_PHOTOS = 10

    try:
        payload     = request.json
        name        = payload.get('name', '').strip()
        images_data = payload.get('images', [])   # list of base64 data URLs

        # Validate name
        if not name:
            return jsonify({'success': False, 'error': 'Name cannot be empty.'}), 400

        if not re.match(r'^[\w\s\-]+$', name):
            return jsonify({'success': False,
                            'error': 'Name contains invalid characters. '
                                     'Use letters, numbers, spaces or hyphens.'}), 400

        # Validate photo count
        if not isinstance(images_data, list) or len(images_data) < MIN_PHOTOS:
            return jsonify({'success': False,
                            'error': f'Please provide at least {MIN_PHOTOS} photos '
                                     f'for accurate recognition. '
                                     f'You sent {len(images_data)}.'}), 400

        # Cap at MAX_PHOTOS (browser should enforce this too, but be safe)
        images_data = images_data[:MAX_PHOTOS]

        # Process each image
        folder_name = name.replace(' ', '_')
        person_dir  = DATASET_DIR / folder_name
        person_dir.mkdir(parents=True, exist_ok=True)

        new_encodings   = []   # encodings successfully extracted from this batch
        saved_count     = 0
        no_face_count   = 0
        multi_face_count = 0

        for idx, image_data in enumerate(images_data):
            if ',' not in image_data:
                continue

            try:
                _, encoded = image_data.split(',', 1)
                img_bytes  = base64.b64decode(encoded)
                nparr      = np.frombuffer(img_bytes, np.uint8)
                frame      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                no_face_count += 1
                continue
            if len(face_locations) > 1:
                multi_face_count += 1
                continue

            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            new_encodings.append(encoding)

            # Save image
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            image_path = person_dir / f"photo_{idx:02d}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            saved_count += 1

        # Require at least MIN_PHOTOS usable face photos
        if saved_count < MIN_PHOTOS:
            # Clean up any partially-saved files
            import shutil
            if person_dir.exists() and not any(person_dir.iterdir()):
                shutil.rmtree(person_dir)

            reasons = []
            if no_face_count:
                reasons.append(f"{no_face_count} photo(s) had no detectable face")
            if multi_face_count:
                reasons.append(f"{multi_face_count} photo(s) had multiple faces")

            detail = ('. ' + '; '.join(reasons) + '.') if reasons else '.'
            return jsonify({
                'success': False,
                'error':   f'Only {saved_count} usable face photos out of '
                           f'{len(images_data)} provided{detail} '
                           f'Please retake with better lighting and only your face in frame.'
            }), 400

        # Duplicate check (compare first new encoding against known set)
        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, new_encodings[0])
            best_idx  = np.argmin(distances)
            if distances[best_idx] < 0.5:
                existing = known_names[best_idx]
                # Remove newly saved folder since it's a duplicate
                import shutil
                shutil.rmtree(person_dir, ignore_errors=True)
                return jsonify({
                    'success': False,
                    'error':   f'This face is already registered as "{existing}".'
                }), 409

        # Hot-reload all new encodings into memory
        for enc in new_encodings:
            known_encodings.append(enc)
            known_names.append(folder_name)

        print(f"[REGISTER] '{folder_name}' registered with {saved_count} photos. "
              f"Total known face encodings: {len(known_encodings)}")

        return jsonify({
            'success': True,
            'message': f'"{name}" registered successfully with {saved_count} photos! '
                       f'Attendance will now be marked automatically.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


# ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def view_logs():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance_logs ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template('logs.html', records=records)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
