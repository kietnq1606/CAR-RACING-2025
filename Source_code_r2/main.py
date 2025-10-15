from client_lib import GetStatus, GetRaw, AVControl, CloseSocket
from Transform import transform_to_tensor
from mobilenet import UnetMobile
import torch
import cv2
import numpy as np
from Controller import *
from Calculate import *
from sign_FSM import *
from get_signage import *
import sys
from pathlib import Path
# ============================================================
# Import YOLOv8 modules
# ============================================================
from ultralytics import YOLO
'''
# ============================================================
# Import YOLOv5 modules
# ============================================================
YOLO_PATH = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(YOLO_PATH))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
'''
# ============================================================
# Constants
# ============================================================
BASE_SPEED = 40
MIN_SPEED = -10
MAX_ANGLE = 45.0 
MIN_ANGLE = -45.0
USE_MORPH = True
USE_AMP = True
# ============================================================
# Main
# ============================================================
sign = set()
if __name__ == "__main__":
    try:
        fsm = signFSM()
        sign_state = None

        # ----------------------------------------------------
        # Load MobileNet model
        # ----------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UnetMobile(in_ch=3, out_ch=1, alpha=0.5)
        model.load_state_dict(torch.load("mobileNetV2_ver4.pth", map_location=device))
        model = model.to(device).eval()
        '''
        # ----------------------------------------------------
        # Load YOLOv5 model (local)
        # ----------------------------------------------------
        model_yolo = DetectMultiBackend("yolo_v2.pt", device=device)
        model_yolo.eval()
        '''
        model_yolo = YOLO("bestSignV4.pt")
        print("✅ Models loaded successfully. Device:", device)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if USE_AMP:
                model.half()

        kernel = np.ones((3, 3), np.uint8) if USE_MORPH else None

        while True:
            state = GetStatus()
            raw_image = GetRaw()
            if raw_image is None:
                continue

            tensor = transform_to_tensor(raw_image).to(device)
            if device.type == "cuda" and USE_AMP:
                tensor = tensor.half()

            with torch.no_grad():
                if device.type == "cuda" and USE_AMP:
                    with torch.cuda.amp.autocast():
                        output = torch.sigmoid(model(tensor)).float().cpu()
                else:
                    output = torch.sigmoid(model(tensor)).cpu()

            segment_image = output.squeeze().numpy()
            binary_mask = (segment_image > 0.5).astype(np.uint8) * 255

            if USE_MORPH:
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            # ----------------------------------------------------
            # YOLO Inference (YOLOv8)
            # ----------------------------------------------------
            results = model_yolo.predict(raw_image, verbose=False, imgsz=320)

            bboxes = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = float(scores[i])
                    cls = int(classes[i])
                    if cls != None:
                        sign.add((cls, conf))
                    bboxes.append((int(x1), int(y1), int(x2), int(y2), conf, cls))

                    if DEBUG_VIEW:
                        cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"ID:{cls} {conf:.2f}"
                        cv2.putText(raw_image, label, (int(x1), max(int(y1) - 5, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            '''
            # ----------------------------------------------------
            # YOLO Inference
            # ----------------------------------------------------
            im = letterbox(raw_image, 320, stride=model_yolo.stride, auto=True)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR -> RGB -> CHW
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            # Forward YOLOpred
            pred = model_yolo(im, augment=False, visualize=False)
            det = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)[0]

            bboxes = []
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], raw_image.shape).round()
                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(cls)
                    conf = float(conf)
                    if conf > sign.get(cls, 0.0):
                        sign[cls] = conf
                    bboxes.append((x1, y1, x2, y2, float(conf), int(cls)))
                    if DEBUG_VIEW:
                        cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID:{int(cls)} {conf:.2f}"
                        cv2.putText(raw_image, label, (x1, max(y1 - 5, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            '''
            # ----------------------------------------------------
            # Car Control Algorithm
            # ----------------------------------------------------
            event = get_sign_event(bboxes, binary_mask)
            fsm.on_sign(event)
            dist = event['dist'] if event is not None else None
            sign_state = fsm.process_sign(dist)

            ang_raw = Ang_error(binary_mask, state['Speed'], sign=sign_state, dis =  dist)
            if sign_state in ("TURN_LEFT", "TURN_RIGHT"):
                angle_filtered = filtered_angle(ang_raw, angle_alpha=0.6)
            else:
                angle_filtered = filtered_angle(ang_raw, angle_alpha=0.25)


            desired_speed = spd_cal(base_spd=BASE_SPEED, angle=angle_filtered,
                                    ang_max=MAX_ANGLE, spd_min=MIN_SPEED, sign=sign_state)

            corrected_angle = pid_control_ang(angle_filtered, state['Angle'])
            corrected_speed = pid_control_spd(desired_speed, state['Speed'])

            if state['Speed'] == 0:
                reset_pid()

            if DEBUG_VIEW:
                print(f"Angle_set:{angle_filtered:.2f} | Angle_cmd:{corrected_angle:.2f} | "
                      f"V_set:{desired_speed:.1f} | V_cmd:{corrected_speed:.1f} | "
                      f"V_meas:{state['Speed']:.1f} | Sign:{sign_state}")

            AVControl(speed=corrected_speed, angle=corrected_angle)

            # ----------------------------------------------------
            # Display (optional, requires GTK)
            # ----------------------------------------------------
            if DEBUG_VIEW:
                # Resize mask to match raw image size
                mask_color = cv2.cvtColor(cv2.resize(binary_mask, (raw_image.shape[1], raw_image.shape[0])), cv2.COLOR_GRAY2BGR)
                # Overlay mask on the raw image
                vis = cv2.addWeighted(raw_image, 0.8, mask_color, 0.4, 0)
                # Show the combined image
                cv2.imshow("YOLO + Lane Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print(sign)
        CloseSocket()
        cv2.destroyAllWindows()
