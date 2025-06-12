# =============== Chuẩn bị dữ liệu ====================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from TransFER_model import StemCNN, LocalCNN, TransFER, FullModel

# Load metadata
df = pd.read_csv('./data/fer2013plus.csv')

# Gốc chứa ảnh
root = Path('./FER_Image')

# Map Usage -> thư mục tương ứng
usage_to_folder = {
    'Training'   : 'FER2013Train',
    'PublicTest' : 'FER2013Valid',
    'PrivateTest': 'FER2013Test'
}

# Ép cột Image name về str
df['Image name'] = df['Image name'].astype(str)

# Tạo cột image_path đầy đủ
df['image_path'] = df.apply(
    lambda r: str(root / usage_to_folder[r['Usage']] / r['Image name']),
    axis=1
)

# 8 nhãn cảm xúc
label_cols = [
    'neutral','happiness','surprise','sadness',
    'anger','disgust','fear','contempt'
]

class FERPlusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.labels = self.df[label_cols].values.astype('float32')

        # Chọn transforms (trực tiếp trên PIL.Image)
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),   # PIL → Tensor
                transforms.Lambda(
                    lambda t: t.repeat(3, 1, 1) if t.size(0) == 1 else t
                ),
                transforms.RandomErasing(p=0.1),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(112),
                transforms.CenterCrop(112),
                transforms.ToTensor(),   # PIL → Tensor
                transforms.Lambda(
                    lambda t: t.repeat(3, 1, 1) if t.size(0) == 1 else t
                ),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 1) Mở file .png dưới dạng PIL Image (grayscale)
        img = Image.open(row['image_path']).convert('L')
        # 2) Áp transform trực tiếp (Resize, ToTensor, v.v.)
        img = self.transform(img)
        # 3) Label
        label = torch.from_numpy(self.labels[idx])
        return img, label


# Tách train / valid / test
train_df = df[df['Usage']=='Training']
valid_df = df[df['Usage']=='PublicTest']
test_df  = df[df['Usage']=='PrivateTest']

# Train model
def train_model(model, train_df, val_df, test_df,
                num_epochs=40, batch_size=64, lr=1e-3,
                milestones=[15,30], gamma=0.1, device='cuda',
                save_path='best_fer_model.pth'):
    """
    - Hiển thị tiến trình train theo phần trăm (%).
    - Lưu mô hình tốt nhất (theo val_acc) vào `save_path`.
    - Sau khi train xong, vẽ biểu đồ train loss, train acc, val acc.
    """
    # 1) Tạo datasets và loaders
    datasets = {
        'train': FERPlusDataset(train_df, train=True),
        'val':   FERPlusDataset(val_df,   train=False),
        'test':  FERPlusDataset(test_df,  train=False),
    }
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == 'train'),
            num_workers=4,
            pin_memory=True
        )
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

            targets = labels.argmax(dim=1).long().to(device)  # [B]

            optimizer.zero_grad()
            outputs = model(imgs)             # [B,8]
            loss = criterion(outputs, targets)  # cross-entropy với integer targets
            
            loss.backward()
            optimizer.step()

            # Cộng dồn loss và correct
            epoch_loss += loss.item() * imgs.size(0)
            epoch_correct += (outputs.argmax(dim=1) == targets).sum().item()
            processed += imgs.size(0)

            # In tiến trình theo %: (batch_idx / n_batches) * 100
            percent = 100 * batch_idx / n_batches
            print(f"\rEpoch {epoch:2d}/{num_epochs} "
                  f"[{processed:5d}/{total_train_samples:5d} "
                  f"({percent:3.0f}%)]  "
                  f"Loss: {loss.item():.4f}", end='')

        # Kết thúc epoch training: tính train_loss, train_acc
        train_loss = epoch_loss / total_train_samples
        train_acc = epoch_correct / total_train_samples

        # 4) Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in loaders['val']:
                imgs = imgs.to(device)
                targets = labels.argmax(dim=1).long().to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(dim=1) == targets).sum().item()
                
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
            imgs = imgs.to(device)
            targets = labels.argmax(dim=1).long().to(device)
            outputs = model(imgs)
            test_correct += (outputs.argmax(dim=1) == targets).sum().item()
    test_acc = test_correct / len(datasets['test'])
    print(f"\nFinal Results - Best Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f}")

    # 7) Vẽ biểu đồ train_loss, train_acc, val_acc
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    # Biểu đồ 1: Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], '-', label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Biểu đồ 2: Training Accuracy và Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_acc'], '-', label='Train Acc', color='orange')
    plt.plot(epochs, history['val_acc'], '-', label='Val Acc', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return model, best_val_acc, test_acc


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Đường dẫn file pretrained IR-50 như đã lưu
    pretrained_path = "./backbone_ir50_ms1m_epoch120.pth"
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
                        num_classes=8,
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
        train_df,
        valid_df,
        test_df,
        num_epochs=40,
        batch_size=64,
        lr=1e-3,
        milestones=[15, 30],
        gamma=0.1,
        device=device,
        save_path='./best_FER_model.pth'
    )