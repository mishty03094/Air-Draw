import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None
color = (20, 150, 255)
drawing_enabled = False

cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Load and resize the paint_frame image
    paint_frame = cv2.imread('_ (2).jpeg')
    if paint_frame is not None:
        paint_frame = cv2.resize(paint_frame, (100, 100))  # Resize to 100x100 pixels

    # Initialize canvas if not done already
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw lines on the canvas
            if 0 <= x <= 100 and 0 <= y <= 100:  # Region of paint_frame
                drawing_enabled = True  # Enable drawing if the paint_frame is touched

                # If drawing is enabled, allow drawing on the canvas
            if drawing_enabled:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, 10)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    # Combine the frame and the canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Overlay the paint_frame as an icon in the top-left corner
    if paint_frame is not None:
        combined[0:100, 0:100] = paint_frame  # Overlay the paint_frame in the top-left corner

    # Display the combined frame
    cv2.imshow("Drawing", combined)

    # Exit the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Clear the canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

# Release resources
cap.release()
cv2.destroyAllWindows()
