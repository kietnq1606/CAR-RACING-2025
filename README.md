# Vision-Based Autonomous Vehicle Control | Round 1

**UIT Car Racing 2025** · Unity Simulation

`Python` · `Fuzzy-PID` · `Feedback Control` · `MobileNetV2` · `PyTorch` · `OpenCV`

## 1. Overview

Hệ thống điều khiển xe tự hành bám làn trong môi trường mô phỏng Unity. Camera cung cấp dữ liệu cho mô hình segmentation; kết quả nhận diện được chuyển thành góc lái và tốc độ mục tiêu cho hai bộ điều khiển phản hồi độc lập.

**Key features:** Fuzzy-PID steering, PID speed control, adaptive look-ahead, speed scheduling và signal filtering.

## 2. System Architecture

| Stage | Input | Processing | Output |
|---|---|---|---|
| Image preprocessing | Camera frame | RGB conversion, resize 320 × 192, normalization | Input tensor |
| Lane segmentation | Input tensor | MobileNetV2-inspired encoder-decoder | Segmentation mask |
| Mask processing | Predicted mask | Sigmoid, threshold 0.5, optional morphological opening/closing | Binary mask |
| Path tracking | Binary mask, vehicle speed | Adaptive look-ahead, geometric tracking, EMA | Steering reference |
| Speed scheduling | Steering reference | Steering-dependent speed adjustment | Speed reference |
| Steering control | Reference and measured angle | Fuzzy-PID | Steering command |
| Speed control | Reference and measured speed | Fixed-gain PID | Speed command |
| Simulation | Steering and speed commands | Unity vehicle simulation | Camera and vehicle feedback |

## 3. Motion Control

### 3.1 Fuzzy-PID Steering Control

Sai số góc lái được tính từ góc lái mục tiêu và góc lái phản hồi của simulator:

$$
e_\theta[k]=\theta_{ref}[k]-\theta_{meas}[k]
$$

Fuzzy Logic sử dụng độ lớn sai số và tốc độ thay đổi sai số để hiệu chỉnh động các hệ số PID thông qua membership functions và rule-based gain scaling.

$$
K_j[k]=K_{j,0}s_j[k],\qquad j\in\{p,i,d\}
$$

Trong đó $s_j[k]$ là hệ số hiệu chỉnh được giới hạn bởi gain scheduler. Bộ PID sử dụng các hệ số mới để tính lệnh góc lái.

### 3.2 PID Speed Control

Bộ điều khiển tốc độ sử dụng hệ số PID cố định và sai số giữa tốc độ mục tiêu và tốc độ phản hồi:

$$
e_v[k]=v_{ref}[k]-v_{meas}[k]
$$

**Controller parameters**

| Controller | Base $K_p$ | Base $K_i$ | Base $K_d$ | Output limits |
|---|---:|---:|---:|---|
| Steering (Fuzzy-PID) | 0.83 | 0.021 | 0.052 | -25 to 25 |
| Speed (PID) | 0.60 | 0.030 | 0.020 | 15 to 50 |

Các giá trị trên là thông số của Round 1; hệ số góc lái thực tế thay đổi theo fuzzy gain scheduler. Giới hạn đầu ra được biểu diễn theo đơn vị của simulator.

### 3.3 Control Refinements

| Technique | Implementation / Purpose |
|---|---|
| Adaptive look-ahead | Chọn hàng ảnh tracking dựa trên tốc độ hiện tại. |
| EMA filtering | Làm mượt góc lái mục tiêu trước khi đưa vào controller. |
| Speed scheduling | Giảm tốc độ mục tiêu khi độ lớn góc lái tăng. |
| Measured loop interval | Sử dụng khoảng thời gian thực giữa các vòng lặp PID. |
| Integral clamping | Giới hạn thành phần tích phân để hạn chế integral windup. |
| IIR derivative filtering | Giảm ảnh hưởng của nhiễu lên thành phần đạo hàm. |
| Output saturation | Giới hạn lệnh điều khiển trong phạm vi đã thiết lập. |

Speed scheduling tạo tốc độ mục tiêu từ 30 đến 50 theo tham số Round 1 (đơn vị simulator), không phải kết quả đo tốc độ thực tế.

## 4. Computer Vision & Path Tracking

- **MobileNetV2-inspired segmentation:** encoder-decoder nhẹ sử dụng inverted residual blocks và depthwise convolutions để tạo mask phục vụ lane tracking.
- **Mask refinement:** threshold 0.5 và tùy chọn morphological opening/closing với kernel 3 × 3.
- **Reference generation:** xác định hai biên trên hàng look-ahead, chọn tracking point lệch phải tại 71.5% chiều rộng vùng nhận diện và tính góc lái theo hình học.
- **Reference smoothing:** EMA với trọng số 0.75 cho giá trị trước và 0.25 cho giá trị mới.
- **Inference optimization:** CUDA và tùy chọn FP16/mixed precision khi GPU tương thích.

## 5. Source Code Organization

| File (under `Source_code/`) | Responsibility |
|---|---|
| `main.py` | Model inference, control loop and simulator commands |
| `Controller.py` | Fuzzy gain scheduling and PID controllers |
| `Calculate.py` | Path tracking, reference filtering and speed scheduling |
| `Transform.py` | Image preprocessing |
| `mobilenet.py` | Segmentation network |

## 6. Getting Started

Requirements: Python, PyTorch, OpenCV, NumPy, Albumentations, competition simulator, compatible client library and `mobileNetV2_ver2.pth` model weights.

```bash
git clone --branch round_1 https://github.com/kietnq1606/CAR-RACING-2025.git
cd CAR-RACING-2025/Source_code
python main.py
```

Khởi động simulator và chuẩn bị client library, model weights trước khi chạy. CUDA là tùy chọn nếu có GPU tương thích.

## 7. Next Round

[Round 2 — Traffic Sign Detection](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_2#readme): mở rộng bài toán nhận diện với YOLO.
