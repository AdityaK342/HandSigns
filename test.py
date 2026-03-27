import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands #used for counting fingers 
mp_drawing = mp.solutions.drawing_utils #used for drawing hands landmarks


hands = mp_hands.Hands() #intialize the hand tracking model

#Helper function
#This function returns the amount of fingers that are currently up based on the hand landmarks provided by MediaPipe.
#pram --> hand_landmarks: the landmarks of the detected hand
#returns --> the number of fingers that are up

def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark

    fingers = 0

    # Index, Middle, Ring, Pinky --> looking at the tip and the joint below it
    tips = [8, 12, 16, 20] 

    for tip in tips:
        if lm[tip].y < lm[tip - 2].y: # if the tip is above the joint, it's considered up
            fingers += 1

    # Thumb (horizontal check)
    if lm[4].x < lm[3].x:
        fingers += 1

    return fingers

#Helper function
#This function returns the basic sign langauge phrases. 
#pram --> hand_landmarks: the landmarks of the detected hand
#returns --> Sign Language gesture as a string (e.g., "FIST", "OPEN PALM", "PEACE", "THUMBS UP", "POINTING", or "UNKNOWN")
def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    fingers = count_fingers(hand_landmarks)

    # basic rules
    if fingers == 0:
        return "FIST"
    elif fingers == 5:
        return "OPEN PALM"
    elif fingers == 2:
        return "PEACE"
    elif fingers == 1:
        # check if thumb specifically is up
        if lm[4].x < lm[3].x:
            return "THUMBS UP"
        else:
            return "POINTING"
    
    return "UNKNOWN"

#opens camera feed --> use q to close
cap = cv2.VideoCapture(0)

# safe-check
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # read one frame from the camera
    ret, frame = cap.read()

    # if frame wasn't captured properly, stop
    if not ret:
        print("Error: Could not read frame.")
        break

    # show the frame in a window
   

    #process the frame with MediaPipe Hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)

            gesture = detect_gesture(hand_landmarks);

            print("Hand Gesture:", gesture)

            # or display on screen
            cv2.putText(frame, f"Hand Gesture: {gesture}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.putText(frame, f"Fingers: {fingers}", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    cv2.imshow("Camera Feed", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#close camera
cap.release()
cv2.destroyAllWindows()