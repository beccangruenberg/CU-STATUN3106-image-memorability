import os
import cv2
import mediapipe as mp
import easyocr
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFilter

input_folder = "lamem_final"

blur_folder = "prototype/final_dataset/blur"
gray_folder = "prototype/final_dataset/grayscale"
crop_folder = "prototype/final_dataset/center_crop"
face_folder = "prototype/final_dataset/face_blur"
text_folder = "prototype/final_dataset/text_blur"

os.makedirs(blur_folder, exist_ok=True)
os.makedirs(gray_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)
os.makedirs(face_folder, exist_ok=True)
os.makedirs(text_folder, exist_ok=True)

# Face detector (MediaPipe)
base_options = python.BaseOptions(model_asset_path="face_detector.tflite")

options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.3
)

face_detector = vision.FaceDetector.create_from_options(options)

# Text detector (EasyOCR)
text_reader = easyocr.Reader(['en'])

for img_name in os.listdir(input_folder):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path)

    # blurred
    blurred = img.filter(ImageFilter.GaussianBlur(5))
    blurred.save(os.path.join(blur_folder, img_name))

    # grayscale
    gray = img.convert("L").convert("RGB")
    gray.save(os.path.join(gray_folder, img_name))

    # cropped
    w, h = img.size
    crop = img.crop((w*0.25, h*0.25, w*0.75, h*0.75))
    crop = crop.resize((w, h))
    crop.save(os.path.join(crop_folder, img_name))

    # face blur
    cv_img = cv2.imread(img_path)

    if cv_img is not None:
        h, w, _ = cv_img.shape

        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        detection_result = face_detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                box = detection.bounding_box

                x = max(0, box.origin_x)
                y = max(0, box.origin_y)
                fw = min(box.width, w - x)
                fh = min(box.height, h - y)

                face_region = cv_img[y:y+fh, x:x+fw]

                if face_region.size > 0:
                    blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
                    cv_img[y:y+fh, x:x+fw] = blurred_face

        cv2.imwrite(os.path.join(face_folder, img_name), cv_img)

    # text blur
    text_img = cv2.imread(img_path)

    if text_img is not None:
        text_results = text_reader.readtext(text_img)

        for bbox, text, confidence in text_results:
            if confidence < 0.3:
                continue

            points = [(int(x), int(y)) for x, y in bbox]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x1 = max(0, min(x_coords))
            y1 = max(0, min(y_coords))
            x2 = min(text_img.shape[1], max(x_coords))
            y2 = min(text_img.shape[0], max(y_coords))

            text_region = text_img[y1:y2, x1:x2]

            if text_region.size > 0:
                blurred_text = cv2.GaussianBlur(text_region, (51, 51), 30)
                text_img[y1:y2, x1:x2] = blurred_text

        cv2.imwrite(os.path.join(text_folder, img_name), text_img)

print("Ablations created")