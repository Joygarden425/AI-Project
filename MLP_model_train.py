import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
import pandas as pd
import joblib

# 훈련 데이터와 테스트 데이터 불러오기
train_data = pd.read_csv('train_combined.csv')
test_data = pd.read_csv('test_combined.csv')

# 훈련 데이터에서 특징과 라벨 분리
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

# 테스트 데이터에서 특징과 라벨 분리
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# MLP 모델 초기화 및 학습
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, verbose=True)
mlp.fit(X_train, y_train)

# 모델 평가
print("훈련 세트 정확도: {:.2f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(mlp.score(X_test, y_test)))

# 모델 저장
model_filename = 'trained_mlp_model.pkl'
joblib.dump(mlp, model_filename)

print(f"모델이 '{model_filename}' 파일로 저장되었습니다.")


train_accuracy = []
test_accuracy = []

# 각 반복에서의 정확도를 기록
for i in range(300):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    train_accuracy.append(accuracy_score(y_train, mlp.predict(X_train)))
    test_accuracy.append(accuracy_score(y_test, mlp.predict(X_test)))

# 손실 값 및 정확도 그래프 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mlp.loss_curve_)
plt.title('Training Loss over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.title('Accuracy over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

# 그래프 저장
plt.savefig('training_progress.png')

# 그래프 표시
plt.show()