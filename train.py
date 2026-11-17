from ultralytics import YOLO
import os

def start_training():
    # 1. Charger le modèle YOLO11 Small
    model = YOLO('yolo11s.pt')

    # 2. Lancer l'entraînement
    results = model.train(
        data='D:/pfa/dataset/data.yaml', 
        epochs=100,           
        imgsz=640,            
        batch=4,              
        device='cpu',         
        workers=2,            
        patience=20,          
        
        # --- RÉGLAGES POUR IMAGES DIFFICILES ---
        mosaic=1.0,           
        mixup=0.1,            
        hsv_v=0.4,            
        hsv_s=0.6,            
        fliplr=0.5,           
        
        # --- DOSSIER DE SORTIE FORCÉ SUR LE DISQUE D ---
        project='D:/pfa/dataset/PFA_Accidents', # <--- MODIFICATION ICI
        name='YOLO11_Small_Light',
        save=True
    )

    print("\n" + "="*50)
    print("FÉLICITATIONS : Entraînement terminé !")
    # Le chemin affiché sera maintenant le bon sur ton disque D
    print(f"Votre modèle final est ici : D:/pfa/dataset/PFA_Accidents/YOLO11_Small_Light/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    if os.path.exists('D:/pfa/dataset/data.yaml'):
        start_training()
    else:
        print("ERREUR : Le fichier D:/pfa/dataset/data.yaml est introuvable !")