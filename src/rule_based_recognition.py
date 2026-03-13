import cv2
from hand_detection import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_con=0.75)
    
    print("Starting Rule-Based Recognition...")
    print("Gestures supported: Rock, Paper, Scissors, Thumbs Up")

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            fingers = detector.fingers_up()
            
            gesture = "Unknown"
            
            # Logic for Rock (All fingers down)
            if fingers == [0, 0, 0, 0, 0]:
                gesture = "Rock"
            # Logic for Paper (All fingers up)
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "Paper"
            # Logic for Scissors (Index and Middle up)
            elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                gesture = "Scissors"
            # Logic for Thumbs Up
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up"

            cv2.putText(img, f"Gesture: {gesture}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Rule-Based Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
