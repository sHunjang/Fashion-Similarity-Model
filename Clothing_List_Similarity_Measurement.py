import os
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from keras.models import load_model
import numpy as np
import cv2  # OpenCV 사용

# YOLOv8 모델 로드
model = YOLO('TOP&BOTTOM_Detection.pt')

# 파인튜닝된 MobileNetV2 모델 로드
fine_tuned_model = load_model('best_model.h5')

# MPS 장치 설정 (사용 가능하면 MPS, 그렇지 않으면 CPU)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# 이미지 전처리 파이프라인 설정
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 색상 히스토그램 추출 함수
def extract_color_histogram(image_array):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])

# 입력된 스타일 이미지를 YOLO로 탐지하여 바운딩 박스 크롭 및 특징 벡터 추출
def extract_feature_vector(image_path):
    # YOLOv8 탐지 수행
    result = model(image_path)
    
    # 원본 이미지 로드
    img = Image.open(image_path)

    # 탐지된 객체의 바운딩 박스 정보를 사용해 이미지를 크롭
    boxes = result[0].boxes  # 첫 번째 이미지에 대한 탐지 결과
    if len(boxes) == 0:
        raise ValueError("탐지된 객체가 없습니다.")

    box = boxes[0]  # 첫 번째 객체만 추출
    xyxy = box.xyxy.cpu().numpy()[0]  # 바운딩 박스 좌표 추출
    cropped_img = img.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
    
    # 이미지가 'RGBA' 모드라면 'RGB'로 변환
    if cropped_img.mode == 'RGBA':
        cropped_img = cropped_img.convert('RGB')
    
    # 크롭된 이미지를 MobileNetV2 모델에 맞게 리사이즈
    resized_img = cropped_img.resize((224, 224))
    
    # 전처리 및 모델에 적용하여 특징 벡터 추출
    input_tensor = preprocess(resized_img).unsqueeze(0).numpy()
    
    # 입력 텐서의 축 순서 변경 (N, C, H, W) -> (N, H, W, C)
    input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))
    
    # OpenCV로 이미지를 numpy 배열로 변환하여 색상 히스토그램 계산
    image_array = np.array(cropped_img)
    color_histogram = extract_color_histogram(image_array)
    
    # 입력 텐서와 색상 히스토그램을 함께 예측 함수에 전달
    feature_vector = fine_tuned_model.predict([input_tensor, np.expand_dims(color_histogram, axis=0)])
    
    # Keras 모델 출력(Numpy 배열)을 PyTorch 텐서로 변환
    return torch.tensor(feature_vector).squeeze()  # (1, N) -> (N,)

# 사용자의 의류 이미지 경로 설정
user_clothing_paths = {
    'top': [
        os.path.join(os.getcwd(), 'User_Clothing_List/Tops/Top1.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Tops/Top2.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Tops/Top3.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Tops/Top4.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Tops/Top5.png'),
    ],
    'bottom': [
        os.path.join(os.getcwd(), 'User_Clothing_List/Bottoms/Bottom1.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Bottoms/Bottom2.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Bottoms/Bottom3.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Bottoms/Bottom4.png'),
        os.path.join(os.getcwd(), 'User_Clothing_List/Bottoms/Bottom5.png'),
    ]
}

# 입력된 스타일 이미지 경로 설정 (비교 대상 스타일)
style_image_path = os.path.join(os.getcwd(), 'Clothings_Combination/test_15.png')

# 입력된 스타일 특징 벡터 추출
style_feature_vector = extract_feature_vector(style_image_path)

# 사용자의 의류에 대한 특징 벡터 추출 함수
def extract_user_clothing_features(clothing_paths):
    clothing_features = []
    for clothing_path in clothing_paths:
        feature_vector = extract_feature_vector(clothing_path)
        clothing_features.append(feature_vector)
    return clothing_features

# 상의 및 하의에 대한 특징 벡터 추출
user_top_features = extract_user_clothing_features(user_clothing_paths['top'])
user_bottom_features = extract_user_clothing_features(user_clothing_paths['bottom'])

# 유사도 측정 함수 (입력된 스타일과 사용자의 의류 비교)
def cosine_similarity(feature, feature_list):
    similarities = []
    for i, other_feature in enumerate(feature_list):
        sim = F.cosine_similarity(
            feature, other_feature, dim=0
        )
        similarities.append((i, sim.item()))
    return similarities

# 선택된 스타일과 사용자의 의류 목록 비교하여 상위 3개 추출
def top_k_similarities(style_feature, user_features, k=3):
    similarities = cosine_similarity(style_feature, user_features)
    similarities.sort(key=lambda x: x[1], reverse=True)  # 유사도 높은 순으로 정렬
    return similarities[:k]  # 상위 k개 반환

# 입력된 스타일과 사용자의 상의/하의 비교하여 상위 3개 출력
top_3_similar_tops = top_k_similarities(style_feature_vector, user_top_features, k=3)
bottom_3_similar_bottoms = top_k_similarities(style_feature_vector, user_bottom_features, k=3)

# 결과 출력
print("입력된 스타일과 유사도가 높은 상위 3개의 상의:")
for idx, sim in top_3_similar_tops:
    print(f"Top_{idx + 1} 유사도: {sim}")

print("입력된 스타일과 유사도가 높은 상위 3개의 하의:")
for idx, sim in bottom_3_similar_bottoms:
    print(f"Bottom_{idx + 1} 유사도: {sim}")

# 유사도가 높은 상의/하의 저장
output_dir = 'similar_clothings'
os.makedirs(output_dir, exist_ok=True)

for i, (idx, sim) in enumerate(top_3_similar_tops):
    img_path = user_clothing_paths['top'][idx]
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, f'similar_top_{i+1}.png'))
    print(f"Top_{idx + 1} 저장됨: similar_top_{i+1}.png")

for i, (idx, sim) in enumerate(bottom_3_similar_bottoms):
    img_path = user_clothing_paths['bottom'][idx]
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, f'similar_bottom_{i+1}.png'))
    print(f"Bottom_{idx + 1} 저장됨: similar_bottom_{i+1}.png")