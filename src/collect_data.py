import cv2
import csv
import os
from hand_detection import HandDetector

def main():
    # Configuration
    gesture_name = input("Enter the gesture name you are collecting data for (e.g., peace, rock, paper): ").lower()
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "gestures.csv")
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=1)
    
    recording = False
    count = 0

    print(f"Press 'r' to start/stop recording landmarks for '{gesture_name}'")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if recording and len(lm_list) != 0:
            # Flatten the landmarks (x, y coordinates for all 21 points)
            # We normalize by making the first landmark (wrist) the origin (0,0)
            base_x, base_y = lm_list[0][1], lm_list[0][2]
            row = []
            for i in range(21):
                row.append(lm_list[i][1] - base_x)
                row.append(lm_list[i][2] - base_y)
            
            # Append label
            row.append(gesture_name)
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            count += 1
            cv2.putText(img, f"Recording... Count: {count}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        status_text = "Status: RECORDING" if recording else "Status: IDLE"
        cv2.putText(img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Gesture: {gesture_name}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Data Collection", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            recording = not recording
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
