import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Concatenate, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from keras.applications import MobileNetV3Small
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# CUDA 장치 설정 (CUDA가 가능하면 GPU 사용, 그렇지 않으면 CPU 사용)
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
else:
    device = '/CPU:0'

print(f"Using device: {device}")

# 하이퍼파라미터 설정
batch_size = 32  # 배치 크기를 더 작게 조정
initial_epochs = 200  # 초기 학습 에포크 수 증가
fine_tune_epochs = 250  # 미세 조정 에포크 수 증가
learning_rate_initial = 1e-6  # 더 작은 초기 학습률로 시작
learning_rate_fine_tune = 1e-6  # 미세 조정 시 학습률

# 데이터 경로 설정
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'
test_dir = 'Dataset/test'

# 데이터 증강 및 로드 (학습, 검증, 테스트)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # 회전 범위 확장
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],  # 밝기 변화 추가
    horizontal_flip=True,
    fill_mode='nearest'  # 빈 공간을 채우는 방식
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# 색상 히스토그램 추출 함수
def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()

    return np.concatenate([h_hist, s_hist, v_hist])

# 배치 단위로 이미지와 색상 히스토그램 생성
def generate_batch_with_color_histograms(data_generator):
    while True:
        images, labels = next(data_generator)
        color_histograms = np.array([extract_color_histogram(image) for image in images])
        yield [images, color_histograms], labels

# 데이터 생성기를 배치 단위로 생성
train_generator_with_hist = generate_batch_with_color_histograms(train_generator)
val_generator_with_hist = generate_batch_with_color_histograms(val_generator)

# 사전 학습된 MobileNetV3Small 모델 로드 (ImageNet weights 사용)
with tf.device(device):
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 모델 헤드 구성
    image_input = base_model.input
    color_input = Input(shape=(768,))  # 256 * 3 채널의 히스토그램

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 색상 히스토그램과 결합
    merged = Concatenate()([x, color_input])

    # 마지막 레이어 구성
    merged = Dense(1024, activation='leaky_relu')(merged)
    merged = Dropout(0.3)(merged)  # Dropout 비율 감소
    predictions = Dense(train_generator.num_classes, activation='softmax')(merged)

    # 모델 정의
    model = Model(inputs=[image_input, color_input], outputs=predictions)

    # 사전 학습된 모델의 일부 레이어만 학습 가능하게 설정
    for layer in base_model.layers[:-10]:  # 마지막 10개 레이어만 학습 가능하게 설정
        layer.trainable = False

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate_initial),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 콜백 설정
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)

    # 학습을 진행하고 history 객체에 저장
    history = model.fit(
        train_generator_with_hist,
        epochs=initial_epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator_with_hist,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, reduce_lr]
    )

    # 미세 조정 단계: 모든 레이어를 학습 가능하게 설정
    for layer in base_model.layers:
        layer.trainable = True

    # 모델 재컴파일
    model.compile(optimizer=Adam(learning_rate=learning_rate_fine_tune),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 재학습 (미세 조정)
    history_finetune = model.fit(
        train_generator_with_hist,
        epochs=fine_tune_epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=val_generator_with_hist,
        validation_steps=val_generator.samples // batch_size,
        callbacks=[checkpoint, reduce_lr]
    )

    # 학습 중 정확도와 손실을 그래프로 시각화 및 저장
    plt.figure(figsize=(12, 4))

    # 1) 학습 손실 및 검증 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history_finetune.history['loss'], label='Fine-tuning Loss')
    plt.plot(history_finetune.history['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 2) 학습 정확도 및 검증 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history_finetune.history['accuracy'], label='Fine-tuning Accuracy')
    plt.plot(history_finetune.history['val_accuracy'], label='Fine-tuning Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 그래프 저장
    graph_output_dir = "training_graphs"
    os.makedirs(graph_output_dir, exist_ok=True)
    plt.savefig(os.path.join(graph_output_dir, "training_accuracy_loss.png"))
    
    # 그래프 창 닫기
    plt.close()

    # 테스트 데이터로 모델 평가
    test_generator_with_hist = generate_batch_with_color_histograms(test_generator)
    test_steps = np.ceil(test_generator.samples / batch_size)
    test_loss, test_acc = model.evaluate(test_generator_with_hist, steps=test_steps)
    print(f'\n테스트 정확도: {test_acc:.4f}')

    # 테스트 데이터 예측 결과 확인
    y_pred = model.predict(test_generator_with_hist, steps=test_steps)
    y_pred_classes = y_pred.argmax(axis=-1)

    # 정답 라벨
    y_true = test_generator.classes[:len(y_pred_classes)]

    # 성능 평가 보고서 출력
    print('\n분류 보고서:\n', classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys()))

    # 혼동 행렬 출력
    print('\n혼동 행렬:\n', confusion_matrix(y_true, y_pred_classes))

    # 최종 모델 저장
    model.save('WOOTD-Model.h5')
