import cv2
import random
import time
import numpy as np
from hand_detection import HandDetector

def draw_futuristic_hud(img, player_score, computer_score, state, countdown=None, result_text=None, player_move=None, computer_move=None):
    h, w, c = img.shape
    overlay = img.copy()

    # 1. Top Bar Background (Semi-transparent dark bar)
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    
    # 2. Bottom Info Bar
    cv2.rectangle(overlay, (0, h-40), (w, h), (20, 20, 20), -1)
    
    # Apply transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # 3. HUD Elements - Neon Lines
    cv2.line(img, (0, 80), (w, 80), (0, 255, 255), 2) # Cyan line
    cv2.line(img, (0, h-40), (w, h-40), (255, 0, 255), 2) # Magenta line

    # 4. Scores
    cv2.putText(img, "PLAYER", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(img, str(player_score), (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.putText(img, "AI SYSTEM", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
    cv2.putText(img, str(computer_score), (w - 180, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.putText(img, "VS", (w // 2 - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 5. Status / Instruction
    if state == "Waiting":
        cv2.putText(img, "PRESS [S] TO INITIALIZE ROUND", (w//2 - 160, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Central Prompt
        cv2.putText(img, "READY TO PLAY?", (w//2 - 140, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    elif state == "Playing":
        cv2.putText(img, "ANALYZING HAND GESTURE...", (w//2 - 120, h - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Countdown Circle
        if countdown is not None and countdown > 0:
            cv2.circle(img, (w//2, h//2), 60, (20, 20, 20), -1)
            cv2.circle(img, (w//2, h//2), 60, (0, 0, 255), 3)
            cv2.putText(img, str(countdown), (w//2 - 20, h//2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    elif state == "Result":
        # Draw Result HUD
        res_overlay = img.copy()
        cv2.rectangle(res_overlay, (w//2 - 200, h//2 - 100), (w//2 + 200, h//2 + 100), (40, 40, 40), -1)
        cv2.addWeighted(res_overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        color = (0, 255, 0) if "Win" in result_text else (0, 0, 255)
        if "Draw" in result_text: color = (255, 255, 0)
        
        cv2.rectangle(img, (w//2 - 200, h//2 - 100), (w//2 + 200, h//2 + 100), color, 3)
        
        cv2.putText(img, result_text, (w//2 - 150, h//2 - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.putText(img, f"YOU: {player_move}", (w//2 - 150, h//2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"AI: {computer_move}", (w//2 - 150, h//2 + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def main():
    cap = cv2.VideoCapture(0)
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    detector = HandDetector(detection_con=0.8)
    
    # Game Variables
    moves = ["Rock", "Paper", "Scissors"]
    state = "Waiting" # Waiting, Playing, Result
    player_move = None
    computer_move = None
    player_score = 0
    computer_score = 0
    countdown_start = 0
    result_text = ""
    result_start = 0

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img, draw=True) # Keep landmarks visible
        lm_list = detector.find_position(img, draw=False)

        countdown_val = None
        if state == "Playing":
            time_elapsed = time.time() - countdown_start
            countdown_val = 3 - int(time_elapsed)

            if countdown_val <= 0:
                # Capture move
                if len(lm_list) != 0:
                    fingers = detector.fingers_up()
                    if fingers == [0, 0, 0, 0, 0]: player_move = "Rock"
                    elif fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]: player_move = "Paper"
                    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]: player_move = "Scissors"
                    else: player_move = "Unknown"

                    if player_move != "Unknown":
                        computer_move = random.choice(moves)
                        if player_move == computer_move:
                            result_text = "DRAW"
                        elif (player_move == "Rock" and computer_move == "Scissors") or \
                             (player_move == "Paper" and computer_move == "Rock") or \
                             (player_move == "Scissors" and computer_move == "Paper"):
                            result_text = "SYSTEM DEFEATED"
                            player_score += 1
                        else:
                            result_text = "AI WINS"
                            computer_score += 1
                        
                        state = "Result"
                        result_start = time.time()
                else:
                    # No hand detected at end of countdown
                    state = "Waiting"

        elif state == "Result":
            if time.time() - result_start > 3:
                state = "Waiting"

        # Apply the Premium HUD
        draw_futuristic_hud(img, player_score, computer_score, state, 
                            countdown=countdown_val, 
                            result_text=result_text, 
                            player_move=player_move, 
                            computer_move=computer_move)

        cv2.imshow("NEURON - Hand Gesture AI", img)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s') and state == "Waiting":
            state = "Playing"
            countdown_start = time.time()
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
