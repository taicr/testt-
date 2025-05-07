**Dự án: Đánh giá mô hình ResNet trên 3 bộ dữ liệu (CIFAR-10, Fashion-MNIST, Tiny ImageNet)**

### Ý tưởng dự án: Phân loại hình ảnh sử dụng ResNet và so sánh hiệu suất trên 3 bộ dữ liệu khác nhau

#### 1. Mô tả dự án
- **Mục tiêu**: Sử dụng kiến trúc **ResNet** (ví dụ: ResNet-18 hoặc ResNet-50) để huấn luyện mô hình phân loại hình ảnh trên 3 bộ dữ liệu: **CIFAR-10**, **Fashion-MNIST**, và **Tiny ImageNet**. Sau đó, đánh giá hiệu suất và chọn bộ dữ liệu/mô hình tốt nhất dựa trên các tiêu chí như độ chính xác, tốc độ, và khả năng tổng quát hóa.
- **Ứng dụng thực tế**: Hiểu cách mô hình deep learning hoạt động trên các loại dữ liệu khác nhau (màu, grayscale, phức tạp), từ đó áp dụng vào các bài toán phân loại thực tế như nhận diện sản phẩm, phân loại đối tượng tự nhiên, hoặc giám sát chất lượng.

#### 2. Ba bộ dữ liệu
Dưới đây là thông tin chi tiết về 3 bộ dữ liệu:

1. **CIFAR-10**:
   - **Mô tả**: Bộ dữ liệu gồm 60,000 hình ảnh màu (RGB, 32x32 pixel) thuộc 10 lớp (xe hơi, chim, mèo, hươu, chó, ếch, ngựa, tàu, máy bay, xe tải).
   - **Đặc điểm**:
     - Hình ảnh màu, kích thước nhỏ, đa dạng đối tượng.
     - Phù hợp cho các mô hình đơn giản hoặc thử nghiệm nhanh.
   - **Số lượng**: 50,000 train, 10,000 test.
   - **Chuẩn bị**: Tải từ torchvision (PyTorch) hoặc TensorFlow Datasets. Chuẩn hóa pixel (0-1), áp dụng augmentation (lật ngang, dịch chuyển).

2. **Fashion-MNIST**:
   - **Mô tả**: Bộ dữ liệu gồm 70,000 hình ảnh thang xám (grayscale, 28x28 pixel) thuộc 10 lớp quần áo (áo thun, quần, áo len, váy, áo khoác, sandal, áo sơ mi, giày thể thao, túi xách, giày cao cổ).
   - **Đặc điểm**:
     - Hình ảnh grayscale (1 kênh), kích thước nhỏ, ít phức tạp hơn CIFAR-10 về màu sắc.
     - Thích hợp cho các bài toán phân loại đơn giản, đặc biệt khi tài nguyên hạn chế.
   - **Số lượng**: 60,000 train, 10,000 test.
   - **Chuẩn bị**: Tải từ torchvision hoặc TensorFlow Datasets. Chuyển thành 3 kênh (lặp lại kênh grayscale) để phù hợp với ResNet. Chuẩn hóa và augmentation.

3. **Tiny ImageNet**:
   - **Mô tả**: Bộ dữ liệu gồm 120,000 hình ảnh màu (RGB, 64x64 pixel) thuộc 200 lớp (các đối tượng tự nhiên như động vật, đồ vật, thực vật). Đây là phiên bản thu nhỏ của ImageNet.
   - **Đặc điểm**:
     - Hình ảnh phức tạp hơn CIFAR-10, số lớp nhiều hơn, yêu cầu mô hình mạnh hơn.
     - Phù hợp để đánh giá khả năng xử lý dữ liệu quy mô lớn.
   - **Số lượng**: 100,000 train, 10,000 validation, 10,000 test.
   - **Chuẩn bị**: Tải từ trang chính Tiny ImageNet. Chuẩn hóa, augmentation (resize về 64x64 hoặc 224x224 nếu cần).

#### 3. Quy trình thực hiện dự án
1. **Chuẩn bị dữ liệu**:
   - Tải 3 bộ dữ liệu từ torchvision (CIFAR-10, Fashion-MNIST) và trang Tiny ImageNet.
   - **Chuẩn hóa**:
     - CIFAR-10: Normalize pixel (0-1), giữ nguyên 32x32.
     - Fashion-MNIST: Chuyển từ 1 kênh sang 3 kênh (lặp kênh grayscale), resize về 32x32 (để đồng nhất với CIFAR-10), normalize.
     - Tiny ImageNet: Resize về 64x64 hoặc 224x224 (tùy cấu hình ResNet), normalize.
   - **Augmentation**: Áp dụng lật ngang, xoay nhẹ, dịch chuyển ngẫu nhiên (sử dụng torchvision.transforms hoặc Albumentations).
   - Chia dữ liệu: Dùng tập train/test mặc định của từng bộ dữ liệu.

2. **Xây dựng mô hình**:
   - **Kiến trúc**: Sử dụng **ResNet-18** (hoặc ResNet-50 nếu có đủ tài nguyên) từ torchvision hoặc TensorFlow.
   - **Điều chỉnh**:
     - CIFAR-10: Thay lớp fully connected cuối thành 10 output (10 lớp).
     - Fashion-MNIST: Tương tự, 10 output.
     - Tiny ImageNet: Thay lớp fully connected thành 200 output.
   - **Huấn luyện**:
     - Framework: PyTorch hoặc TensorFlow.
     - Optimizer: Adam hoặc SGD với momentum.
     - Loss: Cross-Entropy.
     - Batch size: ~128 (tùy GPU).
     - Epoch: ~50-100 (dùng early stopping nếu cần).
     - Learning rate: Bắt đầu 0.001, giảm dần (scheduler).
   - Huấn luyện 3 mô hình riêng biệt trên 3 bộ dữ liệu.

3. **Đánh giá hiệu suất**:
   - **Chỉ số**:
     - **Accuracy (Top-1)**: Tỷ lệ phân loại đúng trên tập test.
     - **Top-5 Accuracy** (cho Tiny ImageNet): Tỷ lệ dự đoán đúng trong top 5 lớp.
     - **FPS (Frames Per Second)**: Đo tốc độ inference trên GPU (ví dụ: RTX 3060).
     - **Loss**: Giá trị loss trên tập test.
   - **Kiểm tra tổng quát hóa**:
     - Thử mô hình CIFAR-10 và Fashion-MNIST trên một số mẫu từ Tiny ImageNet (hoặc ngược lại) để đánh giá khả năng chuyển giao.
   - **Phân tích lỗi**:
     - Sử dụng confusion matrix để xem các lớp bị nhầm lẫn.
     - Visualize các trường hợp phân loại sai (dùng matplotlib hoặc seaborn).

4. **So sánh và chọn mô hình tốt nhất**:
   - **Bảng so sánh**:
     - Accuracy, Top-5 Accuracy (nếu có), FPS, Loss.
     - Thời gian huấn luyện (epoch, giờ).
   - **Phân tích**:
     - **CIFAR-10**: Accuracy cao (~90-95%), FPS nhanh do kích thước ảnh nhỏ, nhưng dữ liệu đơn giản, ít thách thức.
     - **Fashion-MNIST**: Accuracy cao (~90-93%), FPS nhanh nhất (do grayscale), nhưng thiếu thông tin màu sắc, ít phức tạp.
     - **Tiny ImageNet**: Accuracy thấp hơn (~60-70% do 200 lớp), FPS chậm hơn do ảnh lớn và số lớp nhiều, nhưng phản ánh bài toán thực tế hơn.
   - **Tiêu chí chọn**:
     - Nếu ưu tiên độ chính xác và tốc độ: CIFAR-10 hoặc Fashion-MNIST.
     - Nếu ưu tiên khả năng xử lý bài toán phức tạp: Tiny ImageNet.
     - Nếu cần triển khai thực tế: Xem xét FPS và tài nguyên phần cứng.

5. **Cải thiện (Tùy chọn)**:
   - Thử các biến thể ResNet (ResNet-50, ResNet-101) để so sánh.
   - Sử dụng transfer learning: Load pre-trained ResNet (trên ImageNet) và fine-tune trên 3 bộ dữ liệu.
   - Kết hợp dữ liệu (ví dụ: huấn luyện trên CIFAR-10 + Fashion-MNIST) để cải thiện tổng quát hóa.

#### 4. Công cụ và tài nguyên
- **Thư viện**:
  - PyTorch: torchvision (cho ResNet, CIFAR-10, Fashion-MNIST), torch.utils.data.
  - TensorFlow: tf.keras.applications (cho ResNet), tensorflow_datasets.
  - Matplotlib, Seaborn: Visualize kết quả.
- **Phần cứng**: GPU (RTX 3060 hoặc cao hơn) hoặc Google Colab Pro/TPU.
- **Dataset**:
  - CIFAR-10, Fashion-MNIST: Tải từ torchvision hoặc TensorFlow Datasets.
  - Tiny ImageNet: Tải từ https://tiny-imagenet.herokuapp.com/.
- **Môi trường**: Jupyter Notebook, VSCode.

#### 5. Kết quả mong đợi
- **CIFAR-10**: Accuracy ~90-95%, FPS cao (~100-200 FPS trên RTX 3060), dễ huấn luyện.
- **Fashion-MNIST**: Accuracy ~90-93%, FPS cao nhất (~150-250 FPS), ít tốn tài nguyên.
- **Tiny ImageNet**: Accuracy ~60-70% (Top-1), ~85-90% (Top-5), FPS thấp hơn (~50-100 FPS), yêu cầu tính toán lớn.
- **Mô hình tốt nhất**:
  - Nếu ưu tiên tốc độ và độ chính xác cao trên bài toán đơn giản: Fashion-MNIST.
  - Nếu ưu tiên bài toán thực tế, phức tạp: Tiny ImageNet (dù accuracy thấp hơn).
  - CIFAR-10 là lựa chọn cân bằng.

#### 6. Báo cáo và trình bày
- **Báo cáo**:
  - Mô tả 3 bộ dữ liệu, quy trình huấn luyện, đánh giá.
  - Bảng so sánh Accuracy, Top-5 Accuracy, FPS, Loss.
  - Phân tích lỗi (confusion matrix, ví dụ hình ảnh bị nhầm).
  - Kết luận: Bộ dữ liệu nào phù hợp nhất với ResNet và tại sao.
- **Trình bày trực quan**:
  - Biểu đồ Accuracy/Loss theo epoch (train/test).
  - Confusion matrix cho từng bộ dữ liệu.
  - Hiển thị các mẫu phân loại đúng/sai (dùng matplotlib).
  - Biểu đồ so sánh FPS giữa 3 mô hình.

#### 7. Mở rộng (Tùy chọn)
- So sánh ResNet với các kiến trúc khác (VGG, EfficientNet, MobileNet).
- Thử nghiệm trên dữ liệu thực tế (thu thập ảnh từ camera/điện thoại).
- Triển khai mô hình trên thiết bị nhúng (Raspberry Pi, Jetson Nano).
- Dùng kỹ thuật như mixup, cutout để cải thiện accuracy.

---

### Code mẫu (PyTorch)
Dưới đây là code cơ bản để huấn luyện ResNet-18 trên CIFAR-10 (có thể điều chỉnh cho Fashion-MNIST và Tiny ImageNet):

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Model
model = resnet18(pretrained=False).to(device)
model.fc = nn.Linear(model.fc.in_features, 10).to(device)  # 10 classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(50):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')
```

- **Fashion-MNIST**: Thay `CIFAR10` thành `FashionMNIST`, chuyển ảnh 1 kênh sang 3 kênh (dùng `transforms.Lambda`).
- **Tiny ImageNet**: Tải dữ liệu thủ công, điều chỉnh `model.fc` thành 200 output, resize ảnh về 64x64 hoặc 224x224.

---

Dự án này giúp bạn hiểu cách ResNet hoạt động trên các loại dữ liệu khác nhau (màu, grayscale, phức tạp) và đánh giá ưu/nhược điểm của từng bộ dữ liệu. Nếu cần code chi tiết hơn, hướng dẫn tải Tiny ImageNet, hoặc phân tích kết quả, hãy báo mình!
