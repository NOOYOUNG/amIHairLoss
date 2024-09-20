import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# 데이터 경로 설정
train_data_dir = r"C:\deeplearning\amIHairLoss\유형별 두피 이미지\Training"
validation_data_dir = r"C:\deeplearning\amIHairLoss\유형별 두피 이미지\Validation"
categories = ['none', 'mild', 'moderate', 'severe']  # 탈모 단계별 폴더 이름

# 이미지 크기 설정
img_size = 128

def load_images(data_dir):
    data = []
    labels = []
    for category in categories:
        source_path = os.path.join(data_dir, f"Source_{category}")
        label_path = os.path.join(data_dir, f"Label_{category}")

        if not os.path.isdir(source_path):
            print(f"Source folder not found: {source_path}")
            continue
        if not os.path.isdir(label_path):
            print(f"Label folder not found: {label_path}")
            continue

        class_num = categories.index(category)
        
        # 원천 데이터와 라벨 데이터를 모두 로드
        for path in [source_path, label_path]:
            for img_name in os.listdir(path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(path, img_name)
                        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        if img_array is None:
                            print(f"Failed to load image: {img_path}")
                            continue
                        img_array = cv2.resize(img_array, (img_size, img_size))
                        data.append(img_array)
                        labels.append(class_num)
                    except Exception as e:
                        print(e)
    return np.array(data), np.array(labels)

def preprocess_train_data(data_dir):
    X, y = load_images(data_dir)
    if len(X) == 0:
        raise ValueError(f"No images found in directory: {data_dir}")
    X = X / 255.0  # 이미지 정규화
    y = to_categorical(y, num_classes=len(categories))  # One-hot 인코딩
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_val_data(data_dir):
    X, y = load_images(data_dir)
    if len(X) == 0:
        raise ValueError(f"No images found in directory: {data_dir}")
    X = X / 255.0  # 이미지 정규화
    y = to_categorical(y, num_classes=len(categories))  # One-hot 인코딩
    return X, y

# 데이터 전처리 실행
if __name__ == "__main__":
    # 학습 데이터 전처리
    X_train, X_test, y_train, y_test = preprocess_train_data(train_data_dir)
    
    # 검증 데이터 전처리
    X_val, y_val = preprocess_val_data(validation_data_dir)

    # 학습 및 검증 데이터를 저장 (모델 학습 시 사용)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
