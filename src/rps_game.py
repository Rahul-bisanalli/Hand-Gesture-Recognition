import cv2
import random
import time
from hand_detection import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_con=0.75)
    
    # Game Variables
    moves = ["Rock", "Paper", "Scissors"]
    state = "Waiting" # Waiting, Playing, Result
    player_move = None
    computer_move = None
    player_score = 0
    computer_score = 0
    countdown_start = 0

    print("--- ROCK PAPER SCISSORS vs AI ---")
    print("Press 's' to start a round.")
    print("Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)

        # Draw UI
        cv2.putText(img, f"Player: {player_score}  Computer: {computer_score}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if state == "Waiting":
            cv2.putText(img, "Press 's' to Start!", (150, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        elif state == "Playing":
            time_elapsed = time.time() - countdown_start
            countdown = 3 - int(time_elapsed)

            if countdown > 0:
                cv2.putText(img, str(countdown), (280, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
            else:
                # Time's up! Detect player's gesture
                if len(lm_list) != 0:
                    fingers = detector.fingers_up()
                    if fingers == [0, 0, 0, 0, 0]:
                        player_move = "Rock"
                    elif fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]:
                        player_move = "Paper"
                    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                        player_move = "Scissors"
                    else:
                        player_move = "Unknown"

                    if player_move != "Unknown":
                        computer_move = random.choice(moves)
                        
                        # Determine Winner
                        if player_move == computer_move:
                            result = "Draw!"
                        elif (player_move == "Rock" and computer_move == "Scissors") or \
                             (player_move == "Paper" and computer_move == "Rock") or \
                             (player_move == "Scissors" and computer_move == "Paper"):
                            result = "You Win!"
                            player_score += 1
                        else:
                            result = "Computer Wins!"
                            computer_score += 1
                            
                        state = "Result"
                        result_start = time.time()

        elif state == "Result":
            # Display what happened
            cv2.putText(img, f"You: {player_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"AI: {computer_move}", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, result, (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 4)

            # Go back to waiting after 3 seconds
            if time.time() - result_start > 3:
                state = "Waiting"

        cv2.imshow("Rock Paper Scissors Game", img)
        
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
