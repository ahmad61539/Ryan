from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk memberi efek penuaan (simulasi)
def apply_aging_effect(img):
    # Efek blur dan edge
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    edges = cv2.convertScaleAbs(laplacian)
    blended = cv2.addWeighted(blur, 0.7, edges, 0.5, 10)

    # Efek retro (sepia)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(blended.astype(np.float32), sepia_filter)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    # Efek rambut memutih
    white_overlay = np.full_like(sepia_img, (230, 230, 230), dtype=np.uint8)
    alpha = np.linspace(0.4, 0, sepia_img.shape[0]).reshape(-1, 1)
    alpha = np.repeat(alpha, sepia_img.shape[1], axis=1)
    alpha = np.expand_dims(alpha, axis=2)
    white_hair = sepia_img * (1 - alpha) + white_overlay * alpha
    white_hair = white_hair.astype(np.uint8)

    return white_hair


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['photo']
    if file:
        filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg'  
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        img = cv2.imread(path)
        if img is None:
            return "Gagal membaca gambar", 400

        aged_img = apply_aging_effect(img)
        result_path = os.path.join(UPLOAD_FOLDER, 'aged_' + filename)
        cv2.imwrite(result_path, aged_img)

        return send_file(result_path, mimetype='image/jpeg')
    return "Tidak ada file", 400

if __name__ == '__main__':
    app.run(debug=True)
