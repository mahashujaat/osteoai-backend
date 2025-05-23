from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import logging
import json
from scipy.spatial import distance
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
from model import OsteoCNN  # ✅ Import moved model class

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://osteo-ai.vercel.app/:3000", "http://192.168.18.71:3000"]}})
logging.basicConfig(level=logging.DEBUG)

from model import OsteoCNN

# Force load using state_dict instead of entire model
model_path_state_dict = "osteo_cnn_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kl_model = OsteoCNN(num_classes=5).to(device)
kl_model.load_state_dict(torch.load(model_path_state_dict, map_location=device))
kl_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_kl_grade(image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = kl_model(input_tensor)
            return torch.argmax(output, dim=1).item()
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

def split_into_compartments(binary_roi):
    height, width = binary_roi.shape
    top_cutoff = int(0.25 * height)
    bottom_cutoff = int(0.75 * height)
    central_width = int(0.3 * width)
    central_start = (width // 2) - (central_width // 2)
    central_end = (width // 2) + (central_width // 2)

    medial_mask = np.zeros_like(binary_roi, dtype=np.uint8)
    medial_mask[top_cutoff:bottom_cutoff, :central_start] = binary_roi[top_cutoff:bottom_cutoff, :central_start]

    lateral_mask = np.zeros_like(binary_roi, dtype=np.uint8)
    lateral_mask[top_cutoff:bottom_cutoff, central_end:] = binary_roi[top_cutoff:bottom_cutoff, central_end:]

    return medial_mask, lateral_mask

def calc_distance_in_half(roi_half):
    contours, _ = cv2.findContours(roi_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    if len(contours) < 2:
        return None, None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    min_dist = float('inf')
    closest_point1, closest_point2 = None, None

    for point1 in contours[0]:
        for point2 in contours[1]:
            dist = distance.euclidean(point1[0], point2[0])
            if dist < min_dist:
                min_dist = dist
                closest_point1 = tuple(point1[0])
                closest_point2 = tuple(point2[0])

    return min_dist, closest_point1, closest_point2

def find_min_distance(binary_roi):
    medial_roi, lateral_roi = split_into_compartments(binary_roi)
    medial_dist, p1m, p2m = calc_distance_in_half(medial_roi)
    lateral_dist, p1l, p2l = calc_distance_in_half(lateral_roi)

    medial_dist_mm = medial_dist * 0.38 if medial_dist else None
    lateral_dist_mm = lateral_dist * 0.38 if lateral_dist else None

    return medial_dist_mm, lateral_dist_mm, p1m, p2m, p1l, p2l

def process_cropped_roi(cropped_roi, threshold):
    if cropped_roi.shape[0] == 0 or cropped_roi.shape[1] == 0:
        logging.error("Cropped ROI has zero dimensions.")
        return None, None, None

    _, binary_roi = cv2.threshold(cropped_roi, threshold, 255, cv2.THRESH_BINARY)

    if np.all(binary_roi == 0) or np.all(binary_roi == 255):
        logging.error("Binary ROI is entirely black or white.")
        return None, None, None

    try:
        (
            dist_m, dist_l,
            p1m, p2m,
            p1l, p2l,
        ) = find_min_distance(binary_roi)

        min_dist_mm = None
        highlighted_image = None

        if dist_m is not None and (min_dist_mm is None or dist_m < min_dist_mm):
            min_dist_mm = dist_m
            highlighted_image = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
            cv2.circle(highlighted_image, p1m, 5, (0, 0, 255), -1)
            cv2.circle(highlighted_image, p2m, 5, (255, 0, 0), -1)
            cv2.line(highlighted_image, p1m, p2m, (0, 255, 0), 2)

        if dist_l is not None and (min_dist_mm is None or dist_l < min_dist_mm):
            min_dist_mm = dist_l
            highlighted_image = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
            cv2.circle(highlighted_image, p1l, 5, (0, 255, 255), -1)
            cv2.circle(highlighted_image, p2l, 5, (255, 255, 0), -1)
            cv2.line(highlighted_image, p1l, p2l, (255, 0, 255), 2)

        return min_dist_mm, None, highlighted_image

    except Exception as e:
        logging.error(f"Error in process_cropped_roi: {e}")
        return None, None, None

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({'error': 'No image uploaded'}), 400

        threshold = int(request.form.get('threshold', 128))
        image_data = file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        kl_grade = predict_kl_grade(image)
        if kl_grade is None:
            return jsonify({'error': 'Failed to predict KL grade'}), 500

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        background = cv2.GaussianBlur(clahe, (25, 25), 0)
        corrected = cv2.subtract(clahe, background)
        _, binary = cv2.threshold(corrected, threshold, 255, cv2.THRESH_BINARY)

        _, bin_buf = cv2.imencode('.png', binary)
        bin_b64 = base64.b64encode(bin_buf).decode('utf-8')

        _, gray_buf = cv2.imencode('.png', gray)
        gray_b64 = base64.b64encode(gray_buf).decode('utf-8')

        return jsonify({
            'processed_image': bin_b64,
            'grayscale_image': gray_b64,
            'kl_grade_model': kl_grade
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crop-image', methods=['POST'])
def crop_image():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400

        cropped_area = request.form.get('croppedArea')
        if not cropped_area:
            return jsonify({'error': 'No crop area provided'}), 400

        crop = json.loads(cropped_area)
        x, y, w, h = int(crop['x']), int(crop['y']), int(crop['width']), int(crop['height'])
        threshold = int(request.form.get('threshold', 128))

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        cropped = image[y:y+h, x:x+w]
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped

        min_dist, _, highlight = process_cropped_roi(gray_crop, threshold)

        cropped_b64 = base64.b64encode(cv2.imencode('.png', gray_crop)[1]).decode('utf-8')
        highlight_b64 = base64.b64encode(cv2.imencode('.png', highlight)[1]).decode('utf-8') if highlight is not None else None

        return jsonify({
            'cropped_image': cropped_b64,
            'highlighted_image': highlight_b64,
            'min_distance_mm': min_dist
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'OsteoAI backend is running.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
