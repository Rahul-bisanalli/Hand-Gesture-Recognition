import cv2
import time
import numpy as np
import math
from hand_detection import HandDetector
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    # 1. Setup Audio Controller
    devices = AudioUtilities.GetSpeakers()
    volume = devices.EndpointVolume
    
    # Get Volume Range: (Min, Max, Step)
    vol_range = volume.GetVolumeRange()
    min_vol = vol_range[0]
    max_vol = vol_range[1]
    
    # 2. Setup Camera and Detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_con=0.8, max_hands=1)
    
    p_time = 0
    vol_bar = 400
    vol_per = 0

    print("--- NEURON HCI: VOLUME CONTROLLER ---")
    print("Use Thumb and Index finger pinch to control volume.")
    print("Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # 3. Detect Hand
        img = detector.find_hands(img, draw=True)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            # Thumb tip (Point 4) and Index tip (Point 8)
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Draw markers
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

            # 4. Calculate Distance
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Hand range: 25 to 220 (typical gap)
            # Volume range: min_vol to max_vol
            vol = np.interp(length, [25, 200], [min_vol, max_vol])
            vol_bar = np.interp(length, [25, 200], [400, 150])
            vol_per = np.interp(length, [25, 200], [0, 100])
            
            volume.SetMasterVolumeLevel(vol, None)

            # Visual Indicator for distance
            if length < 30:
                cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)

        # 5. Professional HUD Overlay
        # Background bars
        cv2.rectangle(img, (50, 150), (85, 400), (20, 20, 20), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, f'{int(vol_per)}%', (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Title Splash
        cv2.putText(img, "NEURON HCI v1.0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "SYSTEM VOLUME CONTROL", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # FPS Counter (Top Right)
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (w - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("NEURON - HCI Interface", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
