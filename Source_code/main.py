from client_lib import GetStatus, GetRaw, AVControl, CloseSocket
from Transform import transform_to_tensor
from mobilenet import UnetMobile
import torch
import cv2
import numpy as np
from Controller import *
from Calculate import *

# =========================
# Config chung
# =========================        
USE_MORPH = True            # làm sạch mask
USE_AMP = True              # bật AMP(Automatic Mixed Precision) khi có CUDA

# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UnetMobile(in_ch=3, out_ch=1, alpha=0.5)
        model.load_state_dict(torch.load("mobileNetV2_ver2.pth", map_location=device))
        model = model.to(device).eval()
        print("Model loaded. Device:", device)

        # Tối ưu CUDA
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
            #====================================================
            
            segment_image = output.squeeze().numpy()  # HxW, [0..1]
            binary_mask = (segment_image > 0.5).astype(np.uint8) * 255

            if USE_MORPH:
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            #angle and filter noise 
            ang_raw = Ang_error(binary_mask, state['Speed'])
            ang = filtered_angle(ang_raw, angle_alpha=0.25)

            # set target speed according to cornering degree 
            desired_speed = spd_cal(base_spd=50, angle=ang, ang_max=25, spd_min=17)

            # PID
            corrected_angle = pid_control_ang(ang, state['Angle'])
            corrected_speed = pid_control_spd(desired_speed, state['Speed'])

            if DEBUG_VIEW:
                print(f"Angle_set:{ang:.2f}  Angle_cmd:{corrected_angle:.2f}  "
                      f"V_set:{desired_speed:.1f}  V_cmd:{corrected_speed:.1f}  "
                      f"V_meas:{state['Speed']:.1f}")

            AVControl(speed=corrected_speed, angle=corrected_angle)

            if DEBUG_VIEW:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) == 27:  
                    break

    finally:
        CloseSocket()
        cv2.destroyAllWindows()
