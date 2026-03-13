import cv2
import mediapipe as mp
import os
import urllib.request

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Download the official MediaPipe Hand Landmarker model automatically
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "hand_landmarker.task")
        
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker.task from Google... this might take a moment.")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download model: {e}")
                
        # MediaPipe Tasks API (Modern API for Python 3.13)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=float(self.detection_con),
            min_hand_presence_confidence=float(self.track_con),
            min_tracking_confidence=float(self.track_con)
        )
        self.detector = HandLandmarker.create_from_options(options)
        
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.lm_list = []

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # The new API requires an mp.Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hands
        self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                # Custom drawing fallback (since mp.solutions is unavailable)
                connections = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
                               (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15),
                               (13,17), (17,18), (18,19), (19,20), (0,17)]
                
                h, w, c = img.shape
                
                # Draw connection lines
                for connection in connections:
                    idx1, idx2 = connection
                    lm1 = hand_landmarks[idx1]
                    lm2 = hand_landmarks[idx2]
                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    
                # Draw landmark dots
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) != 0:
            # Thumb (right/left check)
            if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

def main():
    import time
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success: break
        
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        
        if len(lm_list) != 0:
            print("Thumb tip coordinates:", lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracking Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
