from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# 모델 로드
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def predict():
    if 'scalp' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['scalp']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 파일을 'io.BytesIO'로 변환
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((128, 128))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 예측
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    print(f"Predictions: {predictions}")
    print(f"Predicted class index: {predicted_class}")

    # 라벨 변환
    class_labels = {0: '양호', 1: '경증', 2: '중등도', 3: '중증'}
    predicted_label = class_labels.get(predicted_class, 'Unknown')

    if predicted_label == 'Unknown':
        return jsonify({'error': 'Predicted class not found in class_labels'}), 500

    # 결과 페이지로 전달
    return render_template('result.html', predicted_class=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)