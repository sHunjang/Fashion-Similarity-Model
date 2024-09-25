import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

# MobileNetV3 모델 정의
class MobileNetWithHist(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetWithHist, self).__init__()
        self.base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()  # 최종 레이어 제거
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(576 + 768, 1024)  # MobileNet output + 히스토그램 크기
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, hist):
        x = self.base_model.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, hist), dim=1)  # 히스토그램과 결합
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 전이 학습을 위한 미세 조정 함수
def fine_tune_model(model, criterion, optimizer, num_epochs, train_loader, val_loader):
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 학습 단계
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            histograms = torch.tensor([extract_color_histogram(inp.permute(1, 2, 0).cpu().numpy()) for inp in inputs]).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs, histograms)
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
                histograms = torch.tensor([extract_color_histogram(inp.permute(1, 2, 0).cpu().numpy()) for inp in inputs]).float().to(device)

                outputs = model(inputs, histograms)
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

        # 최상의 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'fine_tuned_bottoms_model.pth')

    return history

if __name__ == "__main__":

    # MPS 또는 GPU 설정
    try:
        device = torch.device("mps")
        if not torch.backends.mps.is_available():
            raise ValueError("MPS not available")
        print("Using MPS")
    except:
        try:
            device = torch.device("cuda")
            if not torch.cuda.is_available():
                raise ValueError("CUDA not available")
            print("Using CUDA")
        except:
            device = torch.device("cpu")
            print("Using CPU")

    print(f"Using device: {device}")

    # 하이퍼파라미터 설정
    batch_size = 64
    fine_tune_epochs = 50  # 파인튜닝을 위한 에폭 수
    learning_rate_fine_tune = 1e-5
    num_workers = 4  # 데이터 로더에서 사용할 워커 스레드 수 설정

    # 데이터 경로 설정 (하의 데이터셋만 로드)
    train_dir = 'Dataset/bottom_train'
    val_dir = 'Dataset/bottom_val'

    # 데이터 증강 및 로드 (하의 전용 학습, 검증)
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
    }

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

    # 데이터 로더 설정 (num_workers와 pin_memory 추가)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 모델 초기화 (기존 모델 로드 및 마지막 레이어만 학습 가능하도록 수정)
    num_classes = len(train_dataset.classes)
    model = MobileNetWithHist(num_classes).to(device)
    model.load_state_dict(torch.load('WOOTD-Model.pth', map_location=device))

    # 모든 레이어를 고정하고 마지막 레이어만 학습 가능하게 설정
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(1024, num_classes)  # 마지막 레이어 수정
    model.fc2.requires_grad = True  # 마지막 레이어만 학습

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc2.parameters(), lr=learning_rate_fine_tune)

    # 모델 파인튜닝 학습
    history_fine = fine_tune_model(model, criterion, optimizer, fine_tune_epochs, train_loader, val_loader)
