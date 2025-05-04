# Hướng dẫn triển khai hệ thống phân tích ảnh thông minh cho bán lẻ
1. Kiến trúc tổng thể  
Mô hình:  
MNIST (Nhận diện chữ số): Một CNN đơn giản (3-4 tầng convolution) để phân loại chữ số.    
COCO (Phát hiện đối tượng): Mô hình YOLOv8 hoặc Faster R-CNN để phát hiện sản phẩm.  
ImageNet (Phân loại): Mô hình ResNet50 hoặc EfficientNet làm backbone, fine-tune cho phân loại sản phẩm.  
Pipeline:  
Nhận ảnh đầu vào (từ camera hoặc tải lên).  
Chạy mô hình MNIST để nhận diện mã số (nếu có chữ số trong ảnh).  
Chạy mô hình COCO để phát hiện và xác định sản phẩm.  
Chạy mô hình ImageNet để phân loại loại sản phẩm hoặc bối cảnh.  
Kết hợp kết quả để đưa ra báo cáo (ví dụ: “Mã số 1234, 5 chai nước, danh mục đồ uống”).  
2. Công cụ và thư viện  
Thư viện:  
TensorFlow hoặc PyTorch (cho huấn luyện và inference).  
Ultralytics YOLOv8 (cho object detection với COCO).  
Hugging Face Transformers hoặc torchvision (cho mô hình ImageNet).  
OpenCV (xử lý ảnh).  
Môi trường: Google Colab (miễn phí, có GPU), Jupyter Notebook, hoặc máy tính cá nhân với GPU cơ bản (như NVIDIA GTX 1650).  
Triển khai: Dùng Flask hoặc FastAPI để tạo giao diện web đơn giản, cho phép người dùng tải ảnh và xem kết quả.  
## 1. Cài đặt môi trường

```bash
# Tạo môi trường ảo Python
python -m venv retail_vision_env

# Kích hoạt môi trường
# Trên Windows
retail_vision_env\Scripts\activate
# Trên Linux/Mac
source retail_vision_env/bin/activate

# Cài đặt các thư viện cần thiết
pip install tensorflow opencv-python pillow flask ultralytics numpy
```

## 2. Cấu trúc thư mục dự án

```
smart_retail_system/
│
├── app.py                  # Ứng dụng Flask chính
├── models/
│   ├── mnist_model.h5      # Mô hình nhận dạng chữ số
│   └── yolov8n.pt          # Mô hình YOLOv8 (tự động tải về)
├── static/
│   ├── uploads/            # Thư mục lưu ảnh tải lên
│   ├── results/            # Thư mục lưu kết quả phân tích
│   └── css/                # CSS cho giao diện
├── templates/
│   ├── index.html          # Trang chính
│   └── results.html        # Trang hiển thị kết quả
└── utils/
    ├── digit_detector.py   # Module xử lý MNIST 
    ├── object_detector.py  # Module xử lý COCO/YOLO
    └── image_classifier.py # Module xử lý ImageNet
```

## 3. Huấn luyện mô hình MNIST

Tạo file `train_mnist.py` để huấn luyện mô hình nhận dạng chữ số:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Tiền xử lý dữ liệu
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Xây dựng mô hình LeNet
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Biên dịch mô hình
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping để tránh overfit
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Huấn luyện mô hình
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Độ chính xác trên tập test: {test_acc:.4f}')

# Lưu mô hình
model.save('models/mnist_model.h5')
print('Đã lưu mô hình tại models/mnist_model.h5')
```

## 4. Triển khai các module xử lý

### Module nhận dạng chữ số (`utils/digit_detector.py`):

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class DigitDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def preprocess_image(self, image):
        """Tiền xử lý ảnh cho MNIST"""
        # Chuyển sang ảnh xám
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Phát hiện các vùng có thể chứa chữ số
        # Đoạn này có thể phức tạp hơn trong thực tế (sử dụng contour detection)
        regions = self._detect_digit_regions(gray)
        
        results = []
        for region in regions:
            # Tiền xử lý vùng ảnh
            processed = cv2.resize(region, (28, 28))
            processed = processed.astype('float32') / 255.0
            processed = processed.reshape(1, 28, 28, 1)
            
            # Nhận dạng chữ số
            prediction = self.model.predict(processed)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            if confidence > 0.5:  # Ngưỡng tin cậy
                results.append({
                    "digit": int(digit),
                    "confidence": float(confidence)
                })
        
        return results
    
    def _detect_digit_regions(self, gray_image):
        """Phát hiện các vùng có thể chứa chữ số"""
        # Đây là phiên bản đơn giản, trong thực tế sẽ phức tạp hơn
        # Giả sử toàn bộ ảnh là một chữ số
        return [gray_image]

    def detect(self, image):
        """API chính cho việc phát hiện chữ số"""
        return self.preprocess_image(image)
```

### Module phát hiện đối tượng (`utils/object_detector.py`):

```python
import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Ánh xạ danh mục bán lẻ (đơn giản hóa)
        self.retail_categories = {
            "bottle": "Đồ uống",
            "cup": "Đồ uống",
            "wine glass": "Đồ uống",
            "apple": "Thực phẩm",
            "banana": "Thực phẩm",
            "sandwich": "Thực phẩm",
            "orange": "Thực phẩm",
            "broccoli": "Thực phẩm",
            "carrot": "Thực phẩm",
            "hot dog": "Thực phẩm",
            "pizza": "Thực phẩm",
            "donut": "Thực phẩm",
            "cake": "Thực phẩm",
            "chair": "Nội thất",
            "couch": "Nội thất",
            "potted plant": "Nội thất",
            "bed": "Nội thất",
            "laptop": "Điện tử",
            "keyboard": "Điện tử",
            "cell phone": "Điện tử",
            "microwave": "Điện tử",
            "refrigerator": "Điện tử",
            "book": "Văn phòng phẩm",
            "clock": "Đồ gia dụng",
            "vase": "Đồ gia dụng",
            "scissors": "Văn phòng phẩm",
            "toothbrush": "Vật dụng cá nhân",
            "hair drier": "Vật dụng cá nhân",
            "tie": "Thời trang",
            "handbag": "Thời trang",
            "backpack": "Thời trang",
            "umbrella": "Thời trang"
        }
    
    def detect(self, image):
        """Phát hiện đối tượng trong ảnh"""
        results = self.model(image)
        objects = []
        
        # Lấy kết quả nhận diện
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Tọa độ hộp
                confidence = box.conf[0].item()  # Độ tin cậy
                class_id = int(box.cls[0].item())  # ID lớp
                class_name = result.names[class_id]  # Tên lớp
                
                # Chỉ bao gồm các đối tượng trong danh mục bán lẻ và có độ tin cậy cao
                if class_name in self.retail_categories and confidence > self.conf_threshold:
                    objects.append({
                        "class_name": class_name,
                        "category": self.retail_categories.get(class_name, "Khác"),
                        "confidence": float(confidence),
                        "box": [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return objects
```

### Module phân loại ảnh (`utils/image_classifier.py`):

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

class ImageClassifier:
    def __init__(self):
        # Tải mô hình ResNet50 đã được huấn luyện trước trên ImageNet
        self.model = ResNet50(weights='imagenet')
    
    def classify(self, image, top_k=5):
        """Phân loại ảnh sử dụng ResNet50"""
        # Tiền xử lý ảnh cho ResNet50
        img = cv2.resize(image, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Dự đoán
        prediction = self.model.predict(img)
        decoded = decode_predictions(prediction, top=top_k)[0]
        
        # Định dạng kết quả
        classifications = []
        for _, class_name, confidence in decoded:
            classifications.append({
                "class_name": class_name,
                "confidence": float(confidence)
            })
        
        return classifications
```

## 5. Hoàn thiện ứng dụng Flask chính (`app.py`)

```python
import os
import cv2
import time
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for

# Import các module xử lý
from utils.digit_detector import DigitDetector
from utils.object_detector import ObjectDetector
from utils.image_classifier import ImageClassifier

app = Flask(__name__)

# Tạo thư mục cho uploads và results
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Tải các mô hình
print("Đang tải các mô hình...")
digit_detector = DigitDetector('models/mnist_model.h5')
object_detector = ObjectDetector()  # Sẽ tự động tải YOLOv8n
image_classifier = ImageClassifier()  # Sẽ tải ResNet50
print("Đã tải xong các mô hình!")

def analyze_image(image_path):
    """Pipeline phân tích đầy đủ cho một ảnh"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Không thể đọc ảnh"}, None
    
    # Chạy cả ba mô hình
    digits = digit_detector.detect(image)
    objects = object_detector.detect(image)
    classifications = image_classifier.classify(image)
    
    # Kết hợp kết quả
    result = {
        "digits": digits,
        "objects": objects,
        "classifications": classifications
    }
    
    # Hiển thị kết quả (vẽ hộp giới hạn, v.v.)
    visual_result = image.copy()
    
    # Vẽ các đối tượng phát hiện được
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj["box"])
        class_name = obj["class_name"]
        category = obj["category"]
        confidence = obj["confidence"]
        
        cv2.rectangle(visual_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(visual_result, f"{class_name} ({confidence:.2f})", 
                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Nếu tìm thấy các chữ số, hiển thị chúng
    if digits:
        digit_text = "Digits: " + ", ".join([f"{d['digit']} ({d['confidence']:.2f})" for d in digits])
        cv2.putText(visual_result, digit_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Lưu và trả về đường dẫn đến ảnh đã được phân tích
    timestamp = str(int(time.time()))
    result_path = f"static/results/result_{timestamp}.jpg"
    cv2.imwrite(result_path, visual_result)
    
    return result, result_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def upload_and_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có phần file'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn'})
    
    # Lưu ảnh đã tải lên
    filename = f"upload_{int(time.time())}.jpg"
    filepath = os.path.join('static/uploads', filename)
    file.save(filepath)
    
    # Phân tích ảnh
    try:
        analysis_result, result_image_path = analyze_image(filepath)
        if "error" in analysis_result:
            return jsonify(analysis_result)
        
        # Định dạng phản hồi để hiển thị
        summary = {
            'image_path': '/' + filepath,
            'result_image_path': '/' + result_image_path,
            'digit_count': len(analysis_result['digits']),
            'digits': [d['digit'] for d in analysis_result['digits']],
            'object_count': len(analysis_result['objects']),
            'objects': [f"{o['class_name']} ({o['category']})" for o in analysis_result['objects']],
            'classifications': [f"{c['class_name']} ({c['confidence']:.2f})" for c in analysis_result['classifications'][:3]],
            'full_analysis': analysis_result
        }
        
        return jsonify(summary)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
```

## 6. Hoàn thiện giao diện người dùng

### Template chính (`templates/index.html`):

```html
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ thống phân tích ảnh thông minh cho bán lẻ</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .result-container {
            display: none;
            margin-top: 20px;
        }
        .image-container {
            margin-bottom: 20px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .section-title {
            margin-top: 15px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-4">Hệ thống phân tích ảnh thông minh cho bán lẻ</h1>
        
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        Tải lên ảnh để phân tích
                    </div>
                    <div class="card-body">
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="imageUpload" class="form-label">Chọn ảnh:</label>
                                <input class="form-control" type="file" id="imageUpload" name="file" accept="image/*">
                            </div>
                            <button type="submit" class="btn btn-primary">Phân tích</button>
                        </form>
                        
                        <div class="loading" id="loadingIndicator">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Đang tải...</span>
                            </div>
                            <p class="mt-2">Đang phân tích ảnh, vui lòng đợi...</p>
                        </div>
                        
                        <div class="result-container" id="resultContainer">
                            <h3 class="section-title">Kết quả phân tích</h3>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="image-container">
                                        <h4>Ảnh gốc</h4>
                                        <img id="originalImage" src="" alt="Ảnh gốc">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="image-container">
                                        <h4>Ảnh phân tích</h4>
                                        <img id="resultImage" src="" alt="Kết quả">
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-4">
                                <h4 class="section-title">Tóm tắt</h4>
                                <div class="card">
                                    <div class="card-body">
                                        <div id="digitResults" class="mb-3">
                                            <p><strong>Nhận diện chữ số:</strong> <span id="digitValues"></span></p>
                                        </div>
                                        
                                        <div id="objectResults" class="mb-3">
                                            <p><strong>Các đối tượng phát hiện được:</strong></p>
                                            <ul id="objectList"></ul>
                                        </div>
                                        
                                        <div id="classificationResults">
                                            <p><strong>Phân loại ảnh:</strong></p>
                                            <ul id="classificationList"></ul>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loadingIndicator = document.getElementById('loadingIndicator');
            const resultContainer = document.getElementById('resultContainer');
            
            // Hiển thị loading
            loadingIndicator.style.display = 'block';
            resultContainer.style.display = 'none';
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Ẩn loading
                loadingIndicator.style.display = 'none';
                
                if (data.error) {
                    alert('Lỗi: ' + data.error);
                    return;
                }
                
                // Hiển thị kết quả
                document.getElementById('originalImage').src = data.image_path;
                document.getElementById('resultImage').src = data.result_image_path;
                
                // Hiển thị chữ số
                const digitValues = document.getElementById('digitValues');
                if (data.digit_count > 0) {
                    digitValues.textContent = data.digits.join(', ');
                    document.getElementById('digitResults').style.display = 'block';
                } else {
                    digitValues.textContent = 'Không tìm thấy chữ số';
                    document.getElementById('digitResults').style.display = 'block';
                }
                
                // Hiển thị đối tượng
                const objectList = document.getElementById('objectList');
                objectList.innerHTML = '';
                if (data.object_count > 0) {
                    data.objects.forEach(obj => {
                        const li = document.createElement('li');
                        li.textContent = obj;
                        objectList.appendChild(li);
                    });
                    document.getElementById('objectResults').style.display = 'block';
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'Không tìm thấy đối tượng được biết đến';
                    objectList.appendChild(li);
                    document.getElementById('objectResults').style.display = 'block';
                }
                
                // Hiển thị phân loại
                const classificationList = document.getElementById('classificationList');
                classificationList.innerHTML = '';
                if (data.classifications && data.classifications.length > 0) {
                    data.classifications.forEach(cls => {
                        const li = document.createElement('li');
                        li.textContent = cls;
                        classificationList.appendChild(li);
                    });
                    document.getElementById('classificationResults').style.display = 'block';
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'Không thể phân loại';
                    classificationList.appendChild(li);
                    document.getElementById('classificationResults').style.display = 'block';
                }
                
                // Hiển thị container kết quả
                resultContainer.style.display = 'block';
            })
            .catch(error => {
                loadingIndicator.style.display = 'none';
                alert('Lỗi: ' + error);
            });
        });
    </script>
</body>
</html>
```

## 7. Chạy ứng dụng

1. Đảm bảo đã tạo đầy đủ cấu trúc thư mục và các file như đã mô tả
2. Huấn luyện mô hình MNIST:
   ```bash
   python train_mnist.py
   ```
3. Chạy ứng dụng Flask:
   ```bash
   python app.py
   ```
4. Truy cập ứng dụng tại địa chỉ: `http://127.0.0.1:5000`

## 8. Triển khai lên môi trường production (tùy chọn)

Để triển khai ứng dụng lên môi trường production, bạn có thể:

1. Sử dụng Gunicorn + Nginx:
   ```bash
   pip install gunicorn
   
   # Chạy với gunicorn
   gunicorn -w 4 -b 127.0.0.1:8000 app:app
   ```

2. Sử dụng Docker:
   
   Tạo file `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   # Tạo thư mục cần thiết
   RUN mkdir -p static/uploads static/results models
   
   # Đảm bảo huấn luyện mô hình MNIST trước khi đóng gói
   RUN python train_mnist.py
   
   EXPOSE 5000
   
   CMD ["python", "app.py"]
   ```
   
   Tạo file `requirements.txt`:
   ```
   tensorflow
   opencv-python-headless
   pillow
   flask
   ultralytics
   numpy
   ```
   
   Build và chạy Docker container:
   ```bash
   docker build -t smart-retail-vision .
   docker run -p 5000:5000 smart-retail-vision
   ```

## 9. Cải tiến và mở rộng

Sau khi triển khai cơ bản, bạn có thể cải tiến hệ thống bằng cách:

1. **Cải thiện nhận dạng chữ số**:
   - Sử dụng OCR (Optical Character Recognition) thay vì chỉ MNIST để nhận dạng văn bản và mã vạch
   - Tích hợp thư viện như Tesseract OCR hoặc EasyOCR

2. **Cải thiện phát hiện đối tượng**:
   - Fine-tune YOLOv8 trên tập dữ liệu bán lẻ cụ thể
   - Thêm khả năng đếm số lượng sản phẩm tương tự

3. **Nâng cao phân loại**:
   - Fine-tune mô hình ResNet50 trên các danh mục cụ thể của cửa hàng
   - Thêm khả năng phân tích bố cục kệ hàng

4. **Tích hợp phân tích thêm**:
   - Phân tích mật độ khách hàng trong cửa hàng
   - Theo dõi chuyển động và luồng khách hàng
   - Phân tích cảm xúc khách hàng

5. **Nâng cấp giao diện**:
   - Thêm bảng điều khiển thống kê
   - Thêm báo cáo theo thời gian thực
   - Tích hợp với các hệ thống quản lý hàng tồn kho
