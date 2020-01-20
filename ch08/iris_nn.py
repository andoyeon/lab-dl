"""
iris_nn.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('Iris.csv')
print(df.head())

# Id 컬럼 삭제
df = df.iloc[:, 1:]
print(df.head())

# pasndas.DataFrame -> numpy.ndarray로 변환
dataset = df.to_numpy()
print(type(dataset), dataset.shape)

# 데이터와 레이블을 분리
X = dataset[:, :-1].astype('float16')
Y = dataset[:, -1]
print(f'X: {X.shape}, Y: {Y.shape}')
print(X[:5])
print(Y[:5])

# 레이블 데이터 타입을 문자열(setosa, versicolor, virginica)에서 숫자(0, 1, 2)로 변환
encoder = LabelEncoder()  # sklearn.preprocessing
encoder.fit(Y)
Y = encoder.transform(Y)
print(Y[:5])

# one-hot-encoding
Y = to_categorical(Y, 3, dtype='float16')  # tensorflow.keras.utils
print(Y[:5])

# Train/Test 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
print(X_train[:5])
print(Y_train[:5])

# Deep Learning model(NN) 생성
model = Sequential()
# 레이어 추가
model.add(Dense(units=16, activation='relu', input_dim=4))
model.add(Dense(units=3, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, Y_train, batch_size=1, epochs=50,
          validation_data=(X_test, Y_test))

# 모델 평가
eval = model.evaluate(X_test, Y_test)
print(f'Test Loss: {round(eval[0], 4)}, Accuracy: {round(eval[1], 4)}')

# Loss/Accuracy vs Epoch plot
x = range(50)  # epoch
train_loss = history.history['loss']
test_loss = history.history['val_loss']
plt.plot(x, train_loss, c='blue', marker='.', label='Train loss')
plt.plot(x, test_loss, c='red', marker='.', label='Test loss')
plt.legend()
plt.show()

train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
plt.plot(x, train_acc, c='blue', marker='.', label='Train accuracy')
plt.plot(x, test_acc, c='red', marker='.', label='Test accuracy')
plt.legend()
plt.show()

# confusion matrix & classification report
y_true = np.argmax(Y_test, axis=1)
print(y_true)
y_pred = np.argmax(model.predict(X_test), axis=1)
print(y_pred)

cm = confusion_matrix(y_true, y_pred)
print(cm)

report = classification_report(y_true, y_pred)
print(report)
