import cv2
import joblib
import os
import numpy as np
from hand_detection import HandDetector

def main():
    model_path = "models/gesture_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first using train_ml_model.py")
        return

    # Load the trained model
    model = joblib.load(model_path)
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=1)
    
    print("Starting ML Real-Time Prediction...")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        if len(lm_list) != 0:
            # Process landmarks the same way as in collection
            base_x, base_y = lm_list[0][1], lm_list[0][2]
            row = []
            for i in range(21):
                row.append(lm_list[i][1] - base_x)
                row.append(lm_list[i][2] - base_y)
            
            # Predict
            prediction = model.predict([row])[0]
            probability = np.max(model.predict_proba([row]))

            cv2.putText(img, f"Prediction: {prediction} ({probability:.2f})", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ML Prediction", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
