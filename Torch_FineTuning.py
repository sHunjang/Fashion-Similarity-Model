import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Gabor 필터를 사용하여 텍스처 특징 추출 함수 (배치 처리)
def batch_extract_gabor_features(images):
    images = np.uint8(images)
    gabor_filters = []
    ksize = 5  # 커널 크기
    for theta in range(4):  # 0, 45, 90, 135도 방향으로 필터 적용
        theta = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((ksize, ksize), 1.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor_filters.append(kernel)
    
    filtered_images = [cv2.filter2D(image, cv2.CV_8UC3, k) for image in images for k in gabor_filters]
    return np.concatenate([cv2.normalize(f, f).flatten() for f in filtered_images], axis=0)

# 배치 색상 히스토그램 계산 함수
def batch_extract_color_histograms(images):
    histograms = []
    for img in images:
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        histograms.append(np.concatenate([
            cv2.normalize(h_hist, h_hist).flatten(),
            cv2.normalize(s_hist, s_hist).flatten(),
            cv2.normalize(v_hist, v_hist).flatten()
        ]))
    return np.array(histograms)

# MobileNetV3 모델 정의 (Gabor + 히스토그램 결합)
class MobileNetWithTextureHist(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetWithTextureHist, self).__init__()
        self.base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()  # 최종 레이어 제거
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3의 출력 크기, 히스토그램 및 텍스처 크기 합산
        mobilenet_output_size = 576  # MobileNetV3의 출력 크기
        hist_size = 768  # 히스토그램 크기
        texture_size = 602112  # Gabor 필터로 추출된 텍스처 크기

        # fc1의 입력 크기를 603456으로 수정 (576 + 768 + 602112)
        self.fc1 = nn.Linear(mobilenet_output_size + hist_size + texture_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x, hist, texture):
        x = self.base_model.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # [batch_size, mobilenet_output_size]

        # hist와 texture가 2D(batch, feature)가 되도록 reshape
        if hist.dim() == 1:
            hist = hist.unsqueeze(0)  # 차원이 맞지 않으면 배치 차원을 추가
        if texture.dim() == 1:
            texture = texture.unsqueeze(0)

        x = torch.cat((x, hist, texture), dim=1)  # 텍스처, 히스토그램과 결합
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 학습 함수 (히스토그램 및 텍스처를 배치로 처리)
def train_model(model, criterion, optimizer, num_epochs, train_loader, val_loader, patience=10):
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        if early_stop:
            print("조기 종료로 인해 학습을 종료합니다.")
            break

        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 학습 단계
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 배치로 히스토그램과 텍스처 특징 추출
            inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()
            histograms = torch.tensor(batch_extract_color_histograms(inputs_np)).float().to(device)
            textures = torch.tensor(batch_extract_gabor_features(inputs_np)).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs, histograms, textures)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()
                histograms = torch.tensor(batch_extract_color_histograms(inputs_np)).float().to(device)
                textures = torch.tensor(batch_extract_gabor_features(inputs_np)).float().to(device)

                outputs = model(inputs, histograms, textures)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        # 학습 기록
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_acc'].append(val_acc.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 검증 손실이 개선되었는지 확인
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 조기 종료 조건 확인
        if epochs_no_improve >= patience:
            print(f"검증 손실이 {patience} 에포크 동안 개선되지 않아 조기 종료를 실행합니다.")
            early_stop = True

    return history

# 메인 함수
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터 설정
    batch_size = 64
    initial_epochs = 1
    fine_tune_epochs = 1
    learning_rate_initial = 0.01
    learning_rate_fine_tune = 0.01
    num_workers = 6

    # 데이터셋 경로
    train_dir = 'Dataset/train'
    val_dir = 'Dataset/val'
    test_dir = 'Dataset/test'

    # 데이터 증강 및 로드 (학습, 검증, 테스트)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=0.2),
            transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    # 데이터 로더 설정
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 모델 초기화
    num_classes = len(train_dataset.classes)
    model = MobileNetWithTextureHist(num_classes).to(device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_initial)

    # 모델 학습
    history = train_model(model, criterion, optimizer, initial_epochs, train_loader, val_loader)

    # Fine-tuning 단계
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tune)
    history_fine = train_model(model, criterion, optimizer, fine_tune_epochs, train_loader, val_loader)

    # 그래프 시각화 및 저장
    plt.figure(figsize=(12, 4))

    # 1) 학습 손실 및 검증 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.plot(history_fine['train_loss'], label='Fine-tuning Loss')
    plt.plot(history_fine['val_loss'], label='Fine-tuning Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 2) 학습 정확도 및 검증 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history_fine['train_acc'], label='Fine-tuning Accuracy')
    plt.plot(history_fine['val_acc'], label='Fine-tuning Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 그래프 저장
    graph_output_dir = "training_graphs"
    os.makedirs(graph_output_dir, exist_ok=True)
    plt.savefig(os.path.join(graph_output_dir, "training_accuracy_loss.png"))
    plt.close()

    # 테스트 데이터로 모델 평가
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_corrects = 0
    test_total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_np = inputs.permute(0, 2, 3, 1).cpu().numpy()
            histograms = torch.tensor(batch_extract_color_histograms(inputs_np)).float().to(device)
            textures = torch.tensor(batch_extract_gabor_features(inputs_np)).float().to(device)

            outputs = model(inputs, histograms, textures)
            _, preds = torch.max(outputs, 1)

            test_corrects += torch.sum(preds == labels.data)
            test_total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 테스트 정확도 출력
    test_acc = test_corrects.double() / test_total
    print(f'\n테스트 정확도: {test_acc:.4f}')

    # 분류 보고서 및 혼동 행렬 출력
    print('\n분류 보고서:\n', classification_report(y_true, y_pred, target_names=test_dataset.classes))
    print('\n혼동 행렬:\n', confusion_matrix(y_true, y_pred))

    # 최종 모델 저장
    torch.save(model.state_dict(), 'WOOTD-Model_V3Small.pth')
