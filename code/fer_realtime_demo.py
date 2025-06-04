"""
Real-time Facial Emotion Recognition Demo
-----------------------------------------
Sử dụng webcam để phát hiện và căn chỉnh khuôn mặt (MTCNN), 
sau đó dự đoán cảm xúc bằng mô hình FERPlus đã train (best_fer_model.pth).

Bạn cần cài đặt:
  pip install torch torchvision facenet-pytorch opencv-python matplotlib
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------------------------
# 1) MTCNNFaceAligner: detect + align face
# ----------------------------------------
from facenet_pytorch import MTCNN

class MTCNNFaceAligner:
    """
    Sử dụng facenet-pytorch MTCNN để detect và align khuôn mặt.
    - detect_and_align: trả về PIL.Image đã căn chỉnh 112×112 hoặc None nếu không detect được.
    - detect_faces: trả về bounding boxes và xác suất (probs) để vẽ lên khung hình.
    """
    def __init__(self, image_size=112, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        # Khởi tạo MTCNN với size ảnh đầu ra = image_size
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        self.to_pil = transforms.ToPILImage()

    def _prepare_image(self, img_array):
        """
        Chuyển numpy array đầu vào thành PIL Image 3-channel.
        Hỗ trợ:
          - H x W (grayscale)
          - C x H x W
          - 1 x H x W
          - B x C x H x W (lấy batch đầu tiên)
        """
        # Nếu có batch dimension, lấy ảnh đầu tiên
        if img_array.ndim == 4:
            img_array = img_array[0]
        # Nếu grayscale HxW
        if img_array.ndim == 2:
            img_array = img_array[np.newaxis, ...]
        # Nếu shape (1, H, W), nhân đôi thành 3 channels
        if img_array.shape[0] == 1:
            img_array = np.repeat(img_array, 3, axis=0)
        # Chuyển từ [C,H,W] → [H,W,C]
        img_array = np.transpose(img_array, (1, 2, 0))
        img_array = img_array.astype(np.uint8)
        return Image.fromarray(img_array)

    def detect_and_align(self, image):
        """
        Input: PIL.Image hoặc numpy.ndarray (HxW hoặc CxHxW hoặc 1xHxW).
        Trả về: PIL.Image (3x112x112) đã align, hoặc None nếu không detect được mặt.
        """
        if isinstance(image, np.ndarray):
            image = self._prepare_image(image)
        aligned_tensor = self.mtcnn(image)
        if aligned_tensor is None:
            return None
        # aligned_tensor: torch.Tensor shape [3,112,112], giá trị [0,1]
        # chuyển về PIL để thuận tiện hiển thị hoặc predict
        return self.to_pil(aligned_tensor)

    def detect_faces(self, image):
        """
        Input: PIL.Image hoặc numpy.ndarray.
        Trả về: boxes (N×4), probs (N), landmarks (N×5×2)
        Dùng để vẽ bounding box lên frame gốc.
        """
        if isinstance(image, np.ndarray):
            image = self._prepare_image(image)
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        return boxes, probs, landmarks


# ----------------------------------------
# 2) Import định nghĩa mô hình FERPlus
# ----------------------------------------
# Giả định bạn đã lưu toàn bộ module model_MSAD (StemCNN, LocalCNN, TransFER, FullModel) trong `model_msad.py`
from transfer_model import StemCNN, LocalCNN, TransFER, FullModel

# ----------------------------------------
# 3) Nhãn cảm xúc
# ----------------------------------------
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ----------------------------------------
# 4) Hàm load mô hình đã train
# ----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_fer_model(ir50_weights_path, best_model_path):
    """
    Tạo kiến trúc FullModel, load IR-50 pretrained + weights best_model.
    Trả về: model đã set .eval()
    """
    # 1) StemCNN
    stem = StemCNN(ir50_weights_path, device=device).to(device)
    stem.eval()
    # 2) Lấy shape output của stem
    with torch.no_grad():
        dummy = torch.randn(1, 3, 112, 112).to(device)
        feats = stem(dummy)
        C, H, W = feats.shape[1:]
    # 3) LocalCNN
    local = LocalCNN(in_channels=C, num_branches=3, p_drop=0.6).to(device)
    # 4) TransFER
    num_patches = H * W
    transfer = TransFER(in_channels=C,
                        proj_channels=512,
                        num_patches=num_patches,
                        num_classes=len(emotion_labels),
                        depth=8,
                        num_heads=8,
                        mlp_ratio=4,
                        p2=0.3,
                        dropout=0.2).to(device)
    # 5) FullModel
    model = FullModel(stem, local, transfer).to(device)

    # Load weights best_model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# ----------------------------------------
# 5) Tiền xử lý khuôn mặt trước khi predict
# ----------------------------------------
val_transform = transforms.Compose([
    transforms.Resize(112),
    transforms.CenterCrop(112),
    transforms.ToTensor(),  # [0,255] → [0,1]
    transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.size(0) == 1 else t),
])

@torch.no_grad()
def predict_emotion(model, face_img_pil):
    """
    Input: PIL.Image 112×112 (RGB hoặc grayscale)
    Trả về: nhãn và độ tự tin
    """
    x = val_transform(face_img_pil).unsqueeze(0).to(device)  # [1,3,112,112]
    logits = model(x)  # [1,7]
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = np.argmax(probs)
    return emotion_labels[idx], float(probs[idx])

# ----------------------------------------
# 6) Hàm chính demo realtime
# ----------------------------------------
def main():
    # Đường dẫn đến weights
    ir50_pretrained = 'backbone_ir50_ms1m_epoch120.pth'
    best_model_path = "best_fer_model.pth"

    if not os.path.exists(ir50_pretrained):
        raise FileNotFoundError(f"IR50 pretrained not found: {ir50_pretrained}")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"FER best model not found: {best_model_path}")

    print("Loading FERPlus model...")
    model = load_fer_model(ir50_pretrained, best_model_path)
    print("Model loaded. Starting webcam...")

    # Khởi tạo MTCNNFaceAligner
    face_aligner = MTCNNFaceAligner(image_size=112, device=device)

    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Không thể mở webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        # Chuyển từ BGR → RGB để dùng MTCNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect & align face
        aligned = face_aligner.detect_and_align(img_pil)
        if aligned is not None:
            # Dự đoán cảm xúc trên ảnh face đã align
            label, conf = predict_emotion(model, aligned)

            # Lấy bounding box để vẽ
            boxes, probs, _ = face_aligner.detect_faces(img_pil)
            if boxes is not None and len(boxes) > 0:
                # Chọn face đầu tiên (nếu có nhiều face)
                x1, y1, x2, y2 = boxes[0].astype(int)
                # Vẽ bounding box xanh lá
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Hiển thị label + confidence
                text = f"{label}: {conf*100:.1f}%"
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow('Real-time FERPlus Demo', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()