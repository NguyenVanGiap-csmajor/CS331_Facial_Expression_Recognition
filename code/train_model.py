# =============== Chuẩn bị dữ liệu ====================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data.columns = data.columns.str.strip()

pixels = data['pixels'].tolist() # 1
width, height = 48, 48

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')] # 2
    face = np.asarray(face).reshape(width, height) # 3
    
    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.
    # face = face / 255.0 # 4
    # face = cv2.resize(face.astype('uint8'), (width, height)) # 5
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1) # 6

emotions = pd.get_dummies(data['emotion']).to_numpy() # 7

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

class FERDataset(Dataset):
    def __init__(self, X, y, train=True):
        self.X = X.astype(np.uint8)
        self.y = np.argmax(y, axis=1) if y.ndim > 1 else y
        self.train = train
        
        # Optimized transforms
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(128),
                transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.size(0) == 1 else t),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.size(0) == 1 else t),
            ])

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = self.X[idx]
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)  # Remove channel dimension for grayscale
        img = self.transform(img)
        return img, self.y[idx]
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from transfer_model import StemCNN, LocalCNN, TransFER, FullModel


def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                num_epochs=40, batch_size=64, lr=1e-3,
                milestones=[15,30], gamma=0.1, device='cuda',
                save_path='best_fer_model.pth'):
    """
    - Hiển thị tiến trình train theo phần trăm (%).
    - Lưu mô hình tốt nhất (theo val_acc) vào `save_path`.
    - Sau khi train xong, vẽ biểu đồ train loss, train acc, val acc.
    """
    # 1) Tạo datasets và dataloaders
    datasets = {
        'train': FERDataset(X_train, y_train, train=True),
        'val':   FERDataset(X_val,   y_val,   train=False),
        'test':  FERDataset(X_test,  y_test,  train=False),
    }
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name=='train'),
                         num_workers=4, pin_memory=True)
        for name, ds in datasets.items()
    }

    print(f"Dataset sizes - Train: {len(datasets['train'])}, "
          f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    # 2) Thiết lập training
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=gamma)

    best_val_acc = 0.0
    best_model_state = None

    # Dùng để lưu metrics qua các epoch
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # 3) Vòng lặp train
    total_train_samples = len(datasets['train'])
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        processed = 0

        # Tính tổng số batch để hiển thị %
        n_batches = len(loaders['train'])
        for batch_idx, (imgs, labels) in enumerate(loaders['train'], start=1):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Cộng dồn loss và correct
            epoch_loss += loss.item() * imgs.size(0)
            epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
            processed += imgs.size(0)

            # # In tiến trình theo %: (batch_idx / n_batches) * 100
            # percent = 100 * batch_idx / n_batches
            # print(f"\rEpoch {epoch:2d}/{num_epochs} "
            #       f"[{processed:5d}/{total_train_samples:5d} "
            #       f"({percent:3.0f}%)]  "
            #       f"Loss: {loss.item():.4f}", end='')

        # Kết thúc epoch training: tính train_loss, train_acc
        train_loss = epoch_loss / total_train_samples
        train_acc = epoch_correct / total_train_samples

        # 4) Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in loaders['val']:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
        val_acc = val_correct / len(datasets['val'])

        # Lưu vào history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # In kết quả epoch
        print(f"\nEpoch {epoch:2d} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc:.4f} "
              f"| Val Acc: {val_acc:.4f}")

        # Scheduler bước giảm lr
        scheduler.step()
        
        # Lưu state tốt nhất
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, save_path)  # Lưu file mô hình tốt nhất
            print(f"  → New best model saved (val_acc={val_acc:.4f})")

    # 5) Sau khi train xong, nạp lại mô hình tốt nhất
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 6) Kiểm tra trên tập test
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in loaders['test']:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
    test_acc = test_correct / len(datasets['test'])
    print(f"\nFinal Results - Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")

    # 7) Vẽ biểu đồ train_loss, train_acc, val_acc
    epochs = np.arange(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 4))

    # Train Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], '-o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Train Acc
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], '-o', label='Train Acc', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)

    # Val Acc
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_acc'], '-o', label='Val Acc', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model, best_val_acc, test_acc


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Đường dẫn file pretrained IR-50
    pretrained_path = "/kaggle/input/iresnet50/pytorch/default/1/backbone_ir50_ms1m_epoch120.pth"
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained not found: {pretrained_path}")

    # Khởi tạo StemCNN
    stem = StemCNN(pretrained_path, device).to(device)
    with torch.no_grad():
        f = stem(torch.randn(1,3,112,112).to(device))
        C, H, W = f.shape[1:]
    print(f"Feature map: [B, {C}, {H}, {W}]")

    # Khởi tạo LocalCNN
    local = LocalCNN(in_channels=C, num_branches=3, p_drop=0.6).to(device)

    # Khởi tạo TransFER
    num_patches = H * W
    transfer = TransFER(in_channels=C,
                        proj_channels=512,
                        num_patches=num_patches,
                        num_classes=7,
                        depth=8,
                        num_heads=8,
                        mlp_ratio=4,
                        p2=0.3,
                        dropout=0.2).to(device)

    # Kết hợp thành FullModel
    model = FullModel(stem, local, transfer).to(device)
    print("Model initialized successfully!")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Gọi train_model
    model, best_val_acc, test_acc = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        num_epochs=40,
        batch_size=64,
        lr=1e-3,
        milestones=[15, 30],
        gamma=0.1,
        device=device,
        save_path='/kaggle/working/best_fer_model.pth'
    )