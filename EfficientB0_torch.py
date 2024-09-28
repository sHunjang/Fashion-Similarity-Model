import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 장치 설정
device = torch.device("mps") if torch.backends.mps.is_available() else (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print(f"Using device: {device}")

# 하이퍼파라미터 설정
batch_size = 64
initial_epochs = 150
fine_tune_epochs = 100
learning_rate_initial = 1e-4
learning_rate_fine_tune = 1e-6

# 데이터 경로 설정
train_dir = 'Dataset/train'
val_dir = 'Dataset/val'
test_dir = 'Dataset/test'

# 데이터 증강 및 변환
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = val_transform

# 사용자 정의 Dataset 클래스
class CustomDatasetWithHistogram(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def extract_color_histogram(self, image):
        hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        return np.concatenate([h_hist, s_hist, v_hist])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        color_histogram = torch.tensor(self.extract_color_histogram(np.array(image)), dtype=torch.float32)
        return image, color_histogram, label

# 데이터 로드
train_dataset = CustomDatasetWithHistogram(train_dir, transform=train_transform)
val_dataset = CustomDatasetWithHistogram(val_dir, transform=val_transform)
test_dataset = CustomDatasetWithHistogram(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# EfficientNetB0 모델 불러오기 및 수정
class EfficientNetWithHistogram(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithHistogram, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # 분류기 부분 제거
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(1000 + 768, 1024)  # EfficientNet 출력과 히스토그램 결합
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, hist):
        x = self.efficientnet(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        combined = torch.cat((x, hist), dim=1)
        combined = self.fc1(combined)
        combined = self.bn1(combined)
        combined = torch.relu(combined)
        combined = self.dropout(combined)
        output = self.fc2(combined)
        return output

# 모델 생성
num_classes = len(train_dataset.classes)
model = EfficientNetWithHistogram(num_classes=num_classes).to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate_initial)

# 학습 및 평가 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, is_finetune=False):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, histograms, labels in train_loader:
            images, histograms, labels = images.to(device), histograms.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, histograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return history

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, histograms, labels in loader:
            images, histograms, labels = images.to(device), histograms.to(device), labels.to(device)

            outputs = model(images, histograms)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader)
    val_acc = correct / total
    return val_loss, val_acc

# 모델 학습
history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=initial_epochs)

# 미세 조정 (fine-tuning) 단계
for param in model.efficientnet.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tune)
history_finetune = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=fine_tune_epochs)

# 테스트 데이터 평가
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f'\n테스트 정확도: {test_acc:.4f}')

# 테스트 데이터 예측
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, histograms, labels in test_loader:
        images, histograms, labels = images.to(device), histograms.to(device), labels.to(device)
        outputs = model(images, histograms)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# 분류 보고서 출력
print('\n분류 보고서:\n', classification_report(y_true, y_pred, target_names=test_dataset.classes))

# 혼동 행렬 출력
print('\n혼동 행렬:\n', confusion_matrix(y_true, y_pred))

