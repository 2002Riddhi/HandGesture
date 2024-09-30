import cv2
import mediapipe as mp

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)
image_counter = 0

# Function to check if the hand is making a closed fist
def is_fist(hand_landmarks):
    # Hand landmarks index for fingertips (index, middle, ring, pinky) relative to the palm (wrist)
    fingers_tips_ids = [8, 12, 16, 20]  # Corresponds to fingertips of index, middle, ring, pinky
    thumb_tip_id = 4  # Thumb tip landmark

    # Check if all fingertips are below their corresponding lower joints, indicating a closed fist
    for tip_id in fingers_tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            return False

    # Check if the thumb is folded across the palm (close to the index finger's base)
    if hand_landmarks.landmark[thumb_tip_id].x > hand_landmarks.landmark[3].x:
        return False

    return True

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    # Convert frame to RGB since MediaPipe processes images in RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Check if the detected hand is making a closed fist
            if is_fist(hand_landmarks):
                # If fist is detected, display a message
                cv2.putText(frame, "Fist Detected, Capturing Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save the image to the E: drive
                image_name = f"E:/fist_detected_{image_counter}.png"  # Save images in E: drive
                cv2.imwrite(image_name, frame)
                print(f"Fist detected and picture saved as {image_name}!")
                image_counter += 1  # Increment counter for the next picture
            else:
                # Display message when a hand is detected but not a fist
                cv2.putText(frame, "Hand Detected, Waiting for Fist", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        # Display message when no hand is detected
        cv2.putText(frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with hand detection
    cv2.imshow("Hand Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
