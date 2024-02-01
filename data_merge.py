import pandas as pd
import os
import glob

# 데이터셋 경로 설정
dataset_path = ""  # 데이터셋이 저장된 경로
categories = ["train", "test"]
emotions = ["happy", "surprise","neutral"]

# 각 카테고리와 감정에 대해 CSV 파일을 읽고 합치는 함수
def combine_csv_files(category, emotion):
    pattern = os.path.join(dataset_path, f"{category}_{emotion}_batch_*.csv")
    csv_files = glob.glob(pattern)
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    return combined_df

# 각 카테고리와 감정에 대해 데이터를 합치고 저장
for category in categories:
    combined_data = pd.DataFrame()
    for emotion in emotions:
        emotion_data = combine_csv_files(category, emotion)
        combined_data = pd.concat([combined_data, emotion_data], ignore_index=True)
    
    # 합쳐진 데이터를 새로운 CSV 파일로 저장
    combined_data.to_csv(f"{category}_combined.csv", index=False)
