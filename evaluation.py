from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

model = load_model('model.h5')

def evaluate_image(image_path):
    img_array = preprocess_image(image_path)  # image_path를 직접 전달
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predictions: {predictions}")
    print(f"Predicted class index: {predicted_class}")

# 예시로 평가할 이미지 경로
image_path = r'C:\Users\CHOI\Desktop\deeplearning\amIHairLoss\scalp_image\Training\Source_none\0643_A2LEBJJDE00048F_1604379695877_4_LH.jpg' 
evaluate_image(image_path)