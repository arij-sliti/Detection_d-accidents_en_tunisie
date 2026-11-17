from flask import Flask, render_template, request, jsonify, Response, send_file
from ultralytics import YOLO
import cv2
import os
import sqlite3
from datetime import datetime
import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import threading
import time
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # charge les variables du fichier .env

app = Flask(__name__)

# --- CONFIGURATION EMAIL ---
EMAIL_SENDER = "slitiarij11@gmail.com"
EMAIL_PASSWORD = "edzz ntmr xknp fjrh"   # mot de passe d'application
EMAIL_RECIPIENT = "slitiarij11@gmail.com"

# --- CONFIGURATION GROQ (chatbot intelligent) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Chatbot LLM (Groq) prêt")
else:
    print("⚠️ GROQ_API_KEY manquante – le chatbot utilisera le mode dégradé.")
    groq_client = None

# --- CONFIGURATION YOLO ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best.pt')

if not os.path.exists(model_path):
    print("⚠️ best.pt introuvable, utilisation de yolov8n.pt")
    model = YOLO('yolov8n.pt')
else:
    model = YOLO(model_path)

UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = os.path.join(current_dir, 'accidents.db')

# --- Base de données ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    governorate TEXT NOT NULL,
                    uploaded_at TIMESTAMP NOT NULL,
                    accident_count INTEGER DEFAULT 0,
                    is_dangerous INTEGER DEFAULT 0,
                    cause TEXT DEFAULT '',
                    analyzed BOOLEAN DEFAULT 0
                )''')
    try:
        c.execute("ALTER TABLE videos ADD COLUMN cause TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE videos ADD COLUMN analyzed BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id)
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

# --- Alertes par EMAIL ---
active_alerts = {}

def send_email_alert(video_id, location, accident_time, severity):
    if video_id in active_alerts and active_alerts[video_id].get('cancelled', False):
        print(f"Email annulé pour la vidéo {video_id}")
        return
    subject = "🚨 ALERTE ACCIDENT DANGEREUX"
    body = (f"📍 Localisation: {location}\n"
            f"⏰ Heure: {accident_time}\n"
            f"⚠️ Gravité: {severity}")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECIPIENT
            server.send_message(msg)
        print("✅ Email d'alerte envoyé avec succès")
    except Exception as e:
        print(f"❌ Erreur d'envoi email : {e}")
    if video_id in active_alerts:
        del active_alerts[video_id]

def schedule_alert(video_id, governorate, accident_time, severity):
    if video_id in active_alerts and 'timer' in active_alerts[video_id]:
        active_alerts[video_id]['timer'].cancel()
    timer = threading.Timer(10.0, send_email_alert, args=[video_id, governorate, accident_time, severity])
    timer.daemon = True
    timer.start()
    active_alerts[video_id] = {'timer': timer, 'cancelled': False}

@app.route('/api/cancel_alert/<int:video_id>', methods=['POST'])
def cancel_alert(video_id):
    if video_id in active_alerts:
        active_alerts[video_id]['cancelled'] = True
        if 'timer' in active_alerts[video_id]:
            active_alerts[video_id]['timer'].cancel()
        del active_alerts[video_id]
        return jsonify({"success": True, "message": "Alerte annulée"})
    return jsonify({"success": False, "message": "Aucune alerte trouvée"})

# --- Fonctions YOLO ---
def get_accident_count_from_frame(results):
    if results[0].boxes is None:
        return 0
    accident_class_id = 0
    class_ids = results[0].boxes.cls.cpu().numpy()
    return 1 if any(cid == accident_class_id for cid in class_ids) else 0

def infer_cause(results, frame_shape):
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return "Aucun objet détecté"
    boxes = results[0].boxes
    nb_objects = len(boxes)
    conf_mean = boxes.conf.cpu().numpy().mean()
    img_h, img_w = frame_shape[:2]
    surfaces = []
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        surface = (x2 - x1) * (y2 - y1) / (img_w * img_h)
        surfaces.append(surface)
    surf_mean = np.mean(surfaces) if surfaces else 0
    if nb_objects == 1:
        return "Sortie de route / perte de contrôle"
    elif nb_objects == 2:
        if conf_mean > 0.7 and surf_mean > 0.2:
            return "Collision frontale entre deux véhicules"
        else:
            return "Collision légère entre deux véhicules"
    elif nb_objects >= 3:
        if nb_objects >= 5:
            return "Carambolage majeur"
        else:
            return "Carambolage multiple"
    elif surf_mean > 0.4:
        return "Impact violent (véhicules lourds)"
    else:
        return "Cause indéterminée"

def generate_frames(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir {video_path}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT analyzed FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    analyzed = row[0] if row else 0
    conn.close()

    if analyzed:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = model.predict(frame, conf=0.3, imgsz=320, verbose=False)
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()
        return

    accident_detected = False
    frame_count = 0

    def update_accident_info(accident_value, dangerous_value, cause_text):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE videos SET accident_count = ?, is_dangerous = ?, cause = ? WHERE id = ?",
                  (accident_value, dangerous_value, cause_text, video_id))
        conn.commit()
        conn.close()
        print(f"Vidéo {video_id} : accident={accident_value}, dangereux={dangerous_value}, cause={cause_text}")

    def mark_analyzed():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE videos SET analyzed = 1 WHERE id = ?", (video_id,))
        conn.commit()
        conn.close()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        results = model.predict(frame, conf=0.3, imgsz=320, verbose=False)
        has_accident = get_accident_count_from_frame(results)

        if has_accident and not accident_detected:
            accident_detected = True
            conf_accident = 0.0
            if results[0].boxes is not None:
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy()
                for i, cls in enumerate(class_ids):
                    if cls == 0:
                        conf_accident = max(conf_accident, boxes.conf[i].cpu().numpy())
            dangerous = 1 if conf_accident >= 0.31 else 0
            cause = infer_cause(results, frame.shape)
            update_accident_info(1, dangerous, cause)

            if dangerous == 1:
                conn_gov = sqlite3.connect(DB_PATH)
                cur_gov = conn_gov.cursor()
                cur_gov.execute("SELECT governorate FROM videos WHERE id = ?", (video_id,))
                gov_row = cur_gov.fetchone()
                conn_gov.close()
                governorate = gov_row[0] if gov_row else "Inconnu"
                accident_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                severity = "Élevée"
                schedule_alert(video_id, governorate, accident_time, severity)

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    if not accident_detected:
        update_accident_info(0, 0, "")
    mark_analyzed()
    print(f"✅ Vidéo {video_id} terminée - {frame_count} frames traitées")

# --- CHATBOT INTELLIGENT (Groq LLM) ---
@app.route('/api/chatbot/ask', methods=['POST'])
def chatbot_ask():
    data = request.get_json()
    user_question = data.get('question', '').strip()
    if not user_question:
        return jsonify({"answer": "Je n'ai pas compris la question. Pouvez-vous reformuler ?"})
    
    if groq_client is None:
        return jsonify({"answer": "Désolé, le service IA n'est pas configuré. Vérifiez votre clé API Groq."})
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant expert du code de la route tunisien. "
                        "Réponds de manière précise, courte et utile. "
                        "Si tu ne sais pas, dis-le simplement. "
                        "Tu peux citer les articles de loi si nécessaire."
                    )
                },
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
            model="llama-3.3-70b-versatile",  # ou "llama3-8b-8192" selon disponibilité
            temperature=0.5,
            max_tokens=500,
        )
        answer = chat_completion.choices[0].message.content
        
        # Sauvegarde dans l'historique (optionnel)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("INSERT INTO chat_history (question, answer) VALUES (?, ?)", (user_question, answer))
            conn.commit()
            conn.close()
        except Exception as e:
            print("Erreur sauvegarde historique:", e)
        
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Erreur Groq : {e}")
        return jsonify({"answer": "Désolé, le service IA est momentanément indisponible. Veuillez réessayer."})

@app.route('/api/chatbot/history', methods=['GET'])
def chatbot_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, question, answer, timestamp FROM chat_history ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    history = [{'id': r[0], 'question': r[1], 'answer': r[2], 'timestamp': r[3]} for r in rows]
    return jsonify(history)

@app.route('/api/chatbot/history/clear', methods=['DELETE'])
def clear_chat_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# --- FEEDBACK et ML (inchangés) ---
satisfaction_model = None
le_cause = None
le_gov = None
try:
    satisfaction_model = joblib.load('satisfaction_model.pkl')
    le_cause = joblib.load('le_cause.pkl')
    le_gov = joblib.load('le_gov.pkl')
    print("✅ Modèle de satisfaction chargé")
except:
    print("⚠️ Pas de modèle de satisfaction trouvé")

def extract_features_for_prediction(video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT accident_count, is_dangerous, cause, governorate FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    accident_count, is_dangerous, cause, gov = row
    return {
        'accident_count': accident_count,
        'is_dangerous': is_dangerous,
        'cause': cause,
        'governorate': gov
    }

def predict_satisfaction(video_id):
    features = extract_features_for_prediction(video_id)
    if not features or satisfaction_model is None:
        return None
    try:
        cause_enc = le_cause.transform([features['cause']])[0] if features['cause'] in le_cause.classes_ else -1
        gov_enc = le_gov.transform([features['governorate']])[0] if features['governorate'] in le_gov.classes_ else -1
        X = [[features['accident_count'], features['is_dangerous'], cause_enc, gov_enc]]
        proba = satisfaction_model.predict_proba(X)[0][1]
        return round(proba * 100, 1)
    except:
        return None

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    video_id = data.get('video_id')
    rating = data.get('rating')
    comment = data.get('comment', '')
    if not video_id or not rating:
        return jsonify({"error": "Champs manquants"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (video_id, rating, comment) VALUES (?, ?, ?)",
              (video_id, rating, comment))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/api/predict_satisfaction/<int:video_id>')
def api_predict_satisfaction(video_id):
    proba = predict_satisfaction(video_id)
    if proba is None:
        return jsonify({"prediction": None})
    return jsonify({"prediction": proba})

# --- ROUTES PRINCIPALES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/api/accidents_by_date')
def api_accidents_by_date():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, governorate, uploaded_at, accident_count, is_dangerous, cause FROM videos WHERE accident_count > 0 ORDER BY uploaded_at DESC")
    rows = c.fetchall()
    conn.close()
    accidents = [{'id': r[0], 'name': r[1], 'governorate': r[2], 'date': r[3],
                  'accident_count': r[4], 'is_dangerous': r[5], 'cause': r[6] or ''} for r in rows]
    return jsonify(accidents)

@app.route('/map')
def map_view():
    return render_template('map.html')

@app.route('/api/history')
def api_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, governorate, uploaded_at, accident_count, is_dangerous, cause FROM videos ORDER BY uploaded_at DESC")
    rows = c.fetchall()
    conn.close()
    videos = [{'id': r[0], 'name': r[1], 'governorate': r[2], 'uploaded_at': r[3],
               'accident_count': r[4], 'is_dangerous': r[5], 'cause': r[6] or ''} for r in rows]
    return jsonify(videos)

@app.route('/api/delete_video/<int:video_id>', methods=['DELETE'])
def delete_video(video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    if row:
        file_path = row[0]
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Erreur suppression fichier: {e}")
    c.execute("DELETE FROM feedback WHERE video_id = ?", (video_id,))
    c.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

@app.route('/accident-dashboard')
def accident_dashboard():
    xml_path = os.path.join(current_dir, 'cause2026 (1).xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    causes, total_accidents, total_tues, total_blesses = [], 0, 0, 0
    for item in root.findall('item'):
        label = item.find('labelle').text
        accidents = int(item.find('accidents').text)
        tues = int(item.find('tues').text)
        blesses = int(item.find('blesses').text)
        causes.append({'label': label, 'accidents': accidents, 'tues': tues, 'blesses': blesses, 'total_victimes': tues + blesses})
        total_accidents += accidents
        total_tues += tues
        total_blesses += blesses
    causes_sorted = sorted(causes, key=lambda x: x['accidents'], reverse=True)
    top_5 = causes_sorted[:5]
    other_accidents = sum(c['accidents'] for c in causes_sorted[5:])
    labels = [c['label'] for c in causes_sorted]
    accidents_data = [c['accidents'] for c in causes_sorted]
    tues_data = [c['tues'] for c in causes_sorted]
    blesses_data = [c['blesses'] for c in causes_sorted]
    stats = {
        'total_accidents': total_accidents, 'total_tues': total_tues, 'total_blesses': total_blesses,
        'taux_mortalite': round((total_tues / total_accidents) * 100, 2) if total_accidents else 0,
        'moyenne_blesses_par_accident': round(total_blesses / total_accidents, 2) if total_accidents else 0,
        'cause_max_accidents': causes_sorted[0]['label'] if causes_sorted else '',
        'causes': causes_sorted, 'labels_json': labels, 'accidents_json': accidents_data,
        'tues_json': tues_data, 'blesses_json': blesses_data, 'top_5': top_5, 'other_accidents': other_accidents
    }
    return render_template('accident_dashboard.html', stats=stats)

@app.route('/api/statistics')
def api_statistics():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT governorate, SUM(accident_count) FROM videos GROUP BY governorate")
    rows = c.fetchall()
    total_stats = {row[0]: row[1] if row[1] is not None else 0 for row in rows}
    c.execute("SELECT governorate, SUM(accident_count) FROM videos WHERE is_dangerous = 1 GROUP BY governorate")
    rows_danger = c.fetchall()
    dangerous_stats = {row[0]: row[1] if row[1] is not None else 0 for row in rows_danger}
    conn.close()
    return jsonify({"total": total_stats, "dangerous": dangerous_stats})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "Aucun fichier"})
    file = request.files['video']
    video_name = request.form.get('video_name', '').strip()
    governorate = request.form.get('governorate', '').strip()
    if not video_name or not governorate:
        return jsonify({"status": "error", "message": "Nom et gouvernorat requis"})
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO videos (name, file_path, governorate, uploaded_at, accident_count, is_dangerous, cause, analyzed) VALUES (?, ?, ?, ?, 0, 0, '', 0)",
              (video_name, filepath, governorate, datetime.now()))
    video_id = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "video_id": video_id, "file_path": filepath})

@app.route('/video_file/<int:video_id>')
def video_file(video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "Vidéo non trouvée", 404
    return send_file(row[0], mimetype='video/mp4')

@app.route('/video_feed/<int:video_id>')
def video_feed(video_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT file_path FROM videos WHERE id = ?", (video_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return "Vidéo non trouvée", 404
    return Response(generate_frames(row[0], video_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)