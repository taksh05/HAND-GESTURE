import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the hand gesture recognition model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to recognize gestures based on landmarks
def recognize_gesture(hand_landmarks):
    if hand_landmarks:
        finger_tips_ids = [4, 8, 12, 16, 20]
        extended_fingers = 0
        
        for tip_id in finger_tips_ids:
            if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
                extended_fingers += 1
        
        if extended_fingers == 5:
            return "Hello"
        elif extended_fingers == 1:
            return "Thumbs Up"
        elif extended_fingers == 0:
            return "Fist"
        elif extended_fingers == 3:
            return "VICTORY"
        elif extended_fingers == 4:
            return "NICE"
        else:
            return "Gesture not recognized"
    return "No hand detected"

# Start capturing video from the camera
camera_index = 0  # Try changing this if you have multiple cameras
cap = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video capture device at index {camera_index}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand annotations on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Recognize the gesture
            gesture = recognize_gesture(hand_landmarks.landmark)
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()