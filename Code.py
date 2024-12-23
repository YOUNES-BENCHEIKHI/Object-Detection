# How to create a video from images
import cv2
import os

images_folder = r"C:\Users\hp\Desktop\Master S3\Analyse vedio, image medicale\images"

output_video_path = r"C:\Users\hp\Desktop\Master S3\Analyse vedio, image medicale\images\ta_video.mp4"

image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])


if not image_files:
    print("Aucune image trouvée dans le dossier.")
    exit()

first_image_path = os.path.join(images_folder, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

fps = 30  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erreur lors de la lecture de l'image : {image_path}")
        continue

    video_writer.write(image)

# Libérer les ressources
video_writer.release()
print(f"Vidéo créée avec succès : {output_video_path}")
# Detect objects in an image
from ultralytics import YOLO
import cv2
import math

image_path = r"C:\Users\hp\Desktop\Master S3\Analyse vedio, image medicale\archive\No_Apply_Grayscale\No_Apply_Grayscale\Vehicles_Detection.v8i.coco\test\frame_1004_jpg.rf.a6f441255e54620ea11fb3cf34fc0a97.jpg" 
img = cv2.imread(image_path)

if img is None:
    print("Erreur : Impossible de charger l'image.")
    exit()

model = YOLO("yolo-Weights/yolov8n.pt")
print("Modèle YOLO chargé avec succès.")

# Classes d'objets
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

results = model(img)

# Parcourir les résultats de détection
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        confidence = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        # Dessiner un rectangle autour de l'objet détecté et afficher la classe et la confiance
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(img, f"{classNames[cls]} {confidence}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Afficher l'image avec les objets détectés
cv2.imshow("Détection d'objets", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Detect objects in a video
from ultralytics import YOLO
import cv2
import math

# Charger la vidéo
video_path = r"C:\Users\hp\Desktop\Master S3\Analyse vedio, image medicale\archive\Sample_Video_HighQuality.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible de charger la vidéo.")
    exit()

# Charger le modèle
model = YOLO("yolo-Weights/yolov8n.pt")
print("Modèle YOLO chargé avec succès.")

while True:
    success, img = cap.read()  # Lire chaque image de la vidéo
    if not success:
        print("Fin de la vidéo ou erreur de lecture.")
        break

    results = model(img, stream=True)  # Faire la détection avec YOLO

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{classNames[cls]} {confidence}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Détection d'objets", img)

    if cv2.waitKey(1) == ord('q'):  # Appuyer sur 'q' pour quitter
        break

cap.release()
cv2.destroyAllWindows()
#Verify if you can download the video
cap = cv2.VideoCapture(r"C:\Users\hp\Desktop\Master S3\Analyse vedio, image medicale\images\ta_video.mp4")
if not cap.isOpened():
    print("Error: Unable to open the video.")
else:
    ret, frame = cap.read()
    if ret:
        print("Video loaded successfully.")
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
    else:
        print("Error: Unable to read frame.")
cap.release()
#Detect objects using a webcam
from ultralytics import YOLO
import cv2
import math

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush","stylo","pincil"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame. Exiting...")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
           
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

           
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    if img is not None and img.size > 0:
        cv2.imshow('Webcam', img)
    else:
        print("Empty frame detected!")
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

