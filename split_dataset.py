import os
import random
import shutil

# --- CONFIGURATION DES CHEMINS (ADAPTÉE À VOTRE STRUCTURE) ---
# BASE_DIR est le dossier racine de votre projet
BASE_DIR = r"D:\pfa\dataset"

# Dossiers SOURCES (où se trouvent vos 492 images actuellement)
IMAGE_SOURCE = os.path.join(BASE_DIR, "images", "train")
LABEL_SOURCE = os.path.join(BASE_DIR, "label", "train")

# Dossiers DESTINATIONS (où nous allons envoyer les 20%)
IMAGE_VAL = os.path.join(BASE_DIR, "images", "val")
LABEL_VAL = os.path.join(BASE_DIR, "label", "val")

def split_data(percent=0.20):
    # Vérification de l'existence des dossiers sources
    if not os.path.exists(IMAGE_SOURCE):
        print(f"ERREUR : Le dossier source des images est introuvable à : {IMAGE_SOURCE}")
        return
    if not os.path.exists(LABEL_SOURCE):
        print(f"ERREUR : Le dossier source des labels est introuvable à : {LABEL_SOURCE}")
        return

    # Créer les dossiers de destination val s'ils n'existent pas
    os.makedirs(IMAGE_VAL, exist_ok=True)
    os.makedirs(LABEL_VAL, exist_ok=True)

    # Liste de toutes les images présentes dans le dossier train
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    images = [f for f in os.listdir(IMAGE_SOURCE) if f.lower().endswith(image_extensions)]
    
    if not images:
        print(f"Aucune image trouvée dans {IMAGE_SOURCE}")
        return

    # Calcul du nombre d'images à déplacer pour la validation (20%)
    num_val = int(len(images) * percent)
    
    # Sélection aléatoire des échantillons
    val_samples = random.sample(images, num_val)

    print(f"Total images dans train : {len(images)}")
    print(f"Déplacement de {num_val} fichiers vers les dossiers 'val'...")

    for img_name in val_samples:
        # 1. Déplacement de l'image
        src_img = os.path.join(IMAGE_SOURCE, img_name)
        dst_img = os.path.join(IMAGE_VAL, img_name)
        shutil.move(src_img, dst_img)
        
        # 2. Déplacement du label correspondant (.txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(LABEL_SOURCE, label_name)
        dst_label = os.path.join(LABEL_VAL, label_name)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            # Si c'est une image normale sans .txt, on crée un fichier vide dans val
            with open(dst_label, 'w') as f:
                pass

    print("\n--- Séparation terminée avec succès ! ---")
    print(f"Images finales dans TRAIN : {len(os.listdir(IMAGE_SOURCE))}")
    print(f"Images finales dans VAL   : {len(os.listdir(IMAGE_VAL))}")
    print(f"\nVotre fichier data.yaml doit maintenant pointer vers :")
    print(f"train: {os.path.join(BASE_DIR, 'images', 'train')}")
    print(f"val: {os.path.join(BASE_DIR, 'images', 'val')}")

if __name__ == "__main__":
    split_data()