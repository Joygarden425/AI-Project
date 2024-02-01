import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd

# MediaPipe 얼굴 랜드마크 모델 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def calculate_euclidean_distance(point1, point2):
    """ 두 점 사이의 유클리드 거리를 계산합니다. """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_image(image_path):
    """ 이미지에서 얼굴 랜드마크를 탐지하고 유클리드 거리를 계산합니다. """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # 선택된 랜드마크 인덱스
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
    else:
        return None

# 이미지 데이터셋 경로
dataset_path = "archive"
categories = ["train", "test"]
emotions = ["angry", "happy", "sad"]
batch_size = 500

# 나머지 코드는 동일하므로 생략

def count_images_in_directory(directory):
    """ 주어진 디렉토리와 하위 디렉토리에 있는 모든 이미지 파일의 개수를 세는 함수 """
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                count += 1
    return count
# 전체 이미지 개수 계산
total_images = count_images_in_directory(dataset_path)
processed_images = 0

# 각 이미지에 대해 랜드마크 탐지 및 거리 계산
for category in categories:
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, category, emotion)
        images = os.listdir(emotion_path)
        total_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)

        for batch in range(total_batches):
            data = []
            labels = []
            start_index = batch * batch_size
            end_index = start_index + batch_size
            batch_images = images[start_index:end_index]

            for image_name in batch_images:
                image_path = os.path.join(emotion_path, image_name)
                distances = process_image(image_path)

                processed_images += 1
                progress = (processed_images / total_images) * 100
                print(f"Processed {processed_images}/{total_images} images ({progress:.2f}%)")

                if distances is not None:
                    data.append(distances)
                    labels.append(emotion)
                else:
                    print(f"No face landmarks found in {image_path}. Deleting file...")
                    os.remove(image_path)

            # 배치별 데이터프레임 생성 및 저장
            df = pd.DataFrame(data)
            df['label'] = labels
            df.to_csv(f"{category}_{emotion}_batch_{batch+1}.csv", index=False)
