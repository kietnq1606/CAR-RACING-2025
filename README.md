# Vision-Based Autonomous Vehicle Control | Round 2

**UIT Car Racing 2025** · Unity Simulation

`Python` · `YOLO` · `Traffic Sign Detection` · `Computer Vision`

**Baseline:** [Round 1 — System Architecture & Motion Control](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#readme)

## 1. Overview

Round 2 mở rộng bài toán autonomous lane following của Round 1 với **YOLO-based traffic sign detection**. Hệ thống hướng đến việc sử dụng thông tin nhận diện làn đường và biển báo để hỗ trợ điều khiển xe trong môi trường mô phỏng.

README này tập trung vào phần mở rộng; các thuật toán nền tảng được trình bày tại Round 1.

## 2. System Architecture

| Component | Input | Processing | Output |
|---|---|---|---|
| Lane perception (baseline) | Camera frame | [MobileNetV2-based segmentation](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#4-computer-vision--path-tracking) | Lane mask |
| Traffic sign perception (extension) | Camera frame | YOLO object detection | Traffic sign detections |
| Reference generation (baseline) | Lane mask, vehicle state | [Adaptive look-ahead and speed scheduling](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#2-system-architecture) | Steering and speed references |
| Motion control (baseline) | References, vehicle feedback | [Fuzzy-PID steering and PID speed control](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#3-motion-control) | Vehicle control commands |

Các hàng đánh dấu *baseline* mô tả kiến trúc tham chiếu của Round 1. Luồng tích hợp biển báo vào quyết định điều khiển và các thay đổi controller của Round 2 chưa được xác minh trong mã nguồn.

## 3. Traffic Sign Detection — YOLO

YOLO bổ sung khả năng nhận diện biển báo bên cạnh mô hình segmentation dùng cho lane tracking.

| Model | Task | Role |
|---|---|---|
| MobileNetV2-inspired model (baseline) | Semantic segmentation | Xác định vùng phục vụ bám làn. |
| YOLO (extension) | Object detection | Nhận diện biển báo giao thông từ ảnh camera. |

## 4. Motion Control Reference

Các kỹ thuật điều khiển nền tảng được mô tả tại Round 1:

| Technique | Documentation |
|---|---|
| Fuzzy-PID steering & dynamic gain tuning | [Fuzzy-PID Steering Control](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#31-fuzzy-pid-steering-control) |
| PID speed regulation & base gains | [PID Speed Control](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#32-pid-speed-control) |
| Adaptive look-ahead, EMA, anti-windup, IIR filtering | [Control Refinements](https://github.com/kietnq1606/CAR-RACING-2025/tree/round_1#33-control-refinements) |

Không mặc định rằng các tham số PID hoặc phương thức tích hợp của Round 2 giống hoàn toàn Round 1.

## 5. Getting Started

```bash
git clone --branch round_2 https://github.com/kietnq1606/CAR-RACING-2025.git
cd CAR-RACING-2025
```

Chuẩn bị Unity simulator, client library, Python dependencies và các model weights theo yêu cầu của branch. Kiểm tra entry point và cấu hình model trong source code trước khi chạy.
