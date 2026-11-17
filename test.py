from ultralytics import YOLO
import cv2
import os

# 1. Charger le modèle (Chemin relatif : cherche le fichier juste à côté du script)
# On utilise os.path pour être sûr que ça marche sur n'importe quel ordinateur
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best.pt')

model = YOLO(model_path)

# 2. Ouvrir la vidéo
video_path = r'D:\pfa\dataset\video5.mp4'
cap = cv2.VideoCapture(video_path)

print("✅ Modèle chargé localement avec succès !")
print("Touches : 'q' pour quitter, 'Espace' pour mettre en pause.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Détection
    results = model.predict(frame, conf=0.3, imgsz=640, verbose=False)

    # Affichage
    annotated_frame = results[0].plot()
    cv2.imshow("Analyse - PFA (Mode Stable)", annotated_frame)

    # Ralenti (300ms)
    key = cv2.waitKey(300) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord(" "):
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()