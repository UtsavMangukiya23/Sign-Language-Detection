from function import *
import cv2
import numpy as np

# Simple skin color detection fallback (HSV range)
SKIN_LOWER = np.array([0, 30, 60], dtype=np.uint8)
SKIN_UPPER = np.array([20, 150, 255], dtype=np.uint8)

cap = cv2.VideoCapture(0)

cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)

if mp_hands:
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ROI rectangle similar to other scripts
            cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
            image, results = mediapipe_detection(frame, hands)
            draw_styles_landmarks(image, results)

            detected = bool(getattr(results, 'multi_hand_landmarks', None))
            status_text = 'Hand: DETECTED' if detected else 'Hand: NOT DETECTED'
            color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            cv2.imshow('Hand Detection', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
else:
    # Fallback detection using simple skin segmentation
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
        roi = frame[40:400, 0:300]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 1500:  # area threshold to reduce noise
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected = True

        frame[40:400, 0:300] = roi
        status_text = 'Hand: DETECTED' if detected else 'Hand: NOT DETECTED'
        color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow('Hand Detection', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
