import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "dataset"

def get_images_and_labels(path):
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
    ]

    face_samples = []
    ids = []

    for image_path in image_paths:

        pil_img = Image.open(image_path).convert('L')

        img_numpy = np.array(pil_img, 'uint8')

        id = int(os.path.split(image_path)[-1].split(".")[1])

        face_samples.append(img_numpy)
        ids.append(id)

    return face_samples, ids

print("Training faces...")

faces, ids = get_images_and_labels(path)

recognizer.train(faces, np.array(ids))

os.makedirs("trainer", exist_ok=True)

recognizer.write("trainer/trainer.yml")

print("Training Complete!")