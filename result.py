import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# MediaPipe 얼굴 랜드마크 모델 및 드로잉 유틸리티 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

def draw_selected_landmarks(image, landmarks, selected_indices):
    for idx in selected_indices:
        if idx < len(landmarks):
            landmark = landmarks[idx]
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# ... 이전 코드 (함수 정의 등) ...
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
        ##학습시킨 인덱스랑 동일하게 해줘야함
        selected_landmarks_indices = [
            55, 65, 52, 53, 46, 105, 107,                       #왼쪽 눈썹
            285, 295, 282, 283, 276, 334, 336,                  #오른쪽 눈썹
            133, 173, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155, #왼쪽눈
            362, 398, 384, 385, 386, 387, 388, 466, 390, 373, 374, 380, 381, 382, #오른쪽눈
            4,                                                  #코 끝
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,    #상부 입술
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,    #하부 입술
        ]
        selected_landmarks = [landmarks[i] for i in selected_landmarks_indices]

        distances = []
        for i in range(len(selected_landmarks)):
            for j in range(i + 1, len(selected_landmarks)):
                point1 = (selected_landmarks[i].x * width, selected_landmarks[i].y * height)
                point2 = (selected_landmarks[j].x * width, selected_landmarks[j].y * height)
                distance = calculate_euclidean_distance(point1, point2)
                distances.append(distance)
        if len(distances) == 2080:  #학습시킬때 지정한 인덱스 수 n*(n-1)/2 로 지정
            return distances, results
        return None, None
    return []

# 저장된 모델 불러오기
model_filename = 'trained_mlp_model.pkl'
mlp = joblib.load(model_filename)

# 웹캠에서 실시간 얼굴 랜드마크 탐지 및 감정 추론
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
##학습시킨 인덱스랑 동일하게 해줘야함
selected_landmarks_indices = [
            55, 65, 52, 53, 46, 105, 107,                       #왼쪽 눈썹
            285, 295, 282, 283, 276, 334, 336,                  #오른쪽 눈썹
            133, 173, 157, 158, 159, 160, 161, 246, 163, 144, 145, 153, 154, 155, #왼쪽눈
            362, 398, 384, 385, 386, 387, 388, 466, 390, 373, 374, 380, 381, 382, #오른쪽눈
            4,                                                  #코 끝
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,    #상부 입술
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,    #하부 입술
        ]
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        continue

    # 두 번째 창을 위한 검은 배경 이미지 생성
    black_image = np.zeros(image.shape, dtype=np.uint8)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = f"FPS: {fps}"

    distances, results = process_frame(image)
    if distances:
        # 감정 추론
        emotion_prediction = mlp.predict([distances])
        emotion_text = f"Emotion: {emotion_prediction[0]}"
    else:
        emotion_text = "Emotion: Detecting..."

    # 랜드마크 그리기
    # if results and results.multi_face_landmarks:
    #     for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #             black_image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
    #             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
    #             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))
    if results and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw_selected_landmarks(black_image, face_landmarks.landmark, selected_landmarks_indices)

    # FPS 및 감정 표시
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, emotion_text, (image.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # 결과를 화면에 표시
    cv2.imshow('Emotion Recognition', image)
    cv2.imshow('Face Landmarks', black_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
