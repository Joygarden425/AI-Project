import cv2
import mediapipe as mp
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import joblib
import time
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고와 에러만 표시
tf.get_logger().setLevel('ERROR')  # TensorFlow 로거의 레벨을 ERROR로 설정

# MediaPipe 얼굴 랜드마크 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def calculate_euclidean_distance(point1, point2):
    """ 두 점 사이의 유클리드 거리를 계산합니다. """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_frame(image):
    """ 프레임에서 얼굴 랜드마크를 탐지하고 유클리드 거리를 계산합니다. """
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # 선택된 랜드마크 인덱스 (예시)
        selected_landmarks_indices = [
            33, 133, 160, 144, 145, 153, 154, 155, 246, 161, 163, 173, 157, 158, 159, 130, 243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29,  # 오른쪽 눈 주변
            362, 385, 387, 263, 373, 380, 381, 382, 466, 388, 390, 400, 374, 375, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 388, 466, 263,  # 왼쪽 눈 주변
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 124, 156, 70, 63, 105, 66, 107, 55, 65, 52,  # 오른쪽 눈썹
            336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 353, 383, 300, 293, 334, 296, 336, 285, 295, 282,  # 왼쪽 눈썹
            4, 5, 6, 195, 197, 168, 197, 196, 195, 5, 4, 98, 97, 2, 326, 327, 328, 329, 330, 331, 332, 333, 334, 296, 336, 285, 295, 282, 283, 276, 353, 383, 300, 293, 334,  # 코 전체
            61, 291, 0, 17, 13, 312, 78, 95, 88, 178, 87, 14, 317, 402, 318, 14, 87, 178, 88, 95, 78, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183,  # 입술 주변 및 입 주변
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109  # 안면 윤곽
        ]
        selected_landmarks = [landmarks[i] for i in selected_landmarks_indices]

        distances = []
        for i in range(len(selected_landmarks)):
            for j in range(i + 1, len(selected_landmarks)):
                point1 = (selected_landmarks[i].x * width, selected_landmarks[i].y * height)
                point2 = (selected_landmarks[j].x * width, selected_landmarks[j].y * height)
                distance = calculate_euclidean_distance(point1, point2)
                distances.append(distance)
        return distances
    return []

# MLP 모델 불러오기 (학습된 모델을 불러와야 합니다)
# 예시: mlp = load_trained_mlp_model()
# 저장된 모델 불러오기
# 저장된 모델 불러오기
model_filename = 'trained_mlp_model.pkl'
mlp = joblib.load(model_filename)

# 웹캠에서 실시간 얼굴 랜드마크 탐지 및 감정 추론
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = f"FPS: {fps}"

    distances = process_frame(image)
    if distances:
        # 감정 추론
        emotion_prediction = mlp.predict([distances])
        emotion_text = f"Emotion: {emotion_prediction[0]}"
    else:
        emotion_text = "Emotion: Detecting..."

    # FPS 및 감정 표시
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, emotion_text, (image.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # 결과를 화면에 표시
    cv2.imshow('Emotion Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()