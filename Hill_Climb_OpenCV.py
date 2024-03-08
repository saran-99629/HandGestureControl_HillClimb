import cv2 as cv
import mediapipe as mp
from pynput.keyboard import Key, Controller

# Initialize Keyboard Controller
keyboard = Controller()

mp_draw = mp.solutions.drawing_utils # Function to Draw Landmarks over Hand
mp_hand = mp.solutions.hands # Hand Detection Function

fingerTipIds = [4, 8, 12, 16, 20]

# Capturing the Video from the Camera
video = cv.VideoCapture(0)

# Initializing the Hand Detection Function
hands = mp_hand.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

while True:
    success, image = video.read()

    # Converting the Image to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Processing the Image for Hand Detection
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Converting the Image back to BGR
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # List to store the Landmark's Coordinates
    landmarks_list = []

    # If Landmarks Detected i.e., Hand Detected Sucessfully
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[-1]

        for index, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape # Height, Width, Channels
            cx, cy = int(lm.x*w), int(lm.y*h)
            landmarks_list.append([index, cx, cy])

        # Drawing the Landmarks for only One Hand
        # Landmarks will be drawn for the Hand which was Detected First
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

    # Stores 1 if finger is Open and 0 if finger is closed
    fingers_open = []

    if len(landmarks_list) != 0:
        for tipId in fingerTipIds:
            if tipId == 4: # That is the thumb
                if landmarks_list[tipId][1] > landmarks_list[tipId - 1][1]:
                    fingers_open.append(1)
                else: 
                    fingers_open.append(0)
            else:
                if landmarks_list[tipId][2] < landmarks_list[tipId - 2][2]:
                    fingers_open.append(1)
                else: 
                    fingers_open.append(0)

    # Counts the Number of Fingers Open
    count_fingers_open = fingers_open.count(1)

    # If Hand Detected
    if results.multi_hand_landmarks != None:
        # If All Fingers Closed --> Break
        if count_fingers_open == 0:
            cv.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv.FILLED)
            cv.putText(image, "BRAKE", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

            keyboard.press(Key.left)
            keyboard.release(Key.right)

        # If All Fingers Open --> Gas
        elif count_fingers_open == 5:
            cv.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv.FILLED)
            cv.putText(image, "GAS", (45, 375), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

            keyboard.press(Key.right)
            keyboard.release(Key.left)

    # If No Hand Detected
    else:
        keyboard.release(Key.right)
        keyboard.release(Key.left)

    # Show the Video
    cv.imshow("Frame", image)
    
    # Close the Video if "q" key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()