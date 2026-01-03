import cv2
import os

cap = cv2.VideoCapture(0)

directory = 'Image'
os.makedirs(os.path.join(directory, 'A'), exist_ok=True)
os.makedirs(os.path.join(directory, 'B'), exist_ok=True)
os.makedirs(os.path.join(directory, 'C'), exist_ok=True)

def next_index(label_folder: str) -> int:
    files = [f for f in os.listdir(label_folder) if f.lower().endswith('.png')]
    return len(files)

# Burst capture state
burst_label = None  # 'A' | 'B' | 'C' | None
burst_remaining = 0
burst_start = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    roi = frame[40:400, 0:300]

    # Overlay UI text
    cv2.putText(frame, 'Press a/b/c for 30-frame burst, q to quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if burst_label:
        cv2.putText(frame, f'Capturing {burst_label}: {burst_remaining} left', (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('data', frame)
    cv2.imshow('RoI', roi)

    # If in burst mode, save current ROI and decrement
    if burst_label and burst_remaining > 0:
        label_dir = os.path.join(directory, burst_label)
        idx = burst_start + (30 - burst_remaining)
        # zero-pad to keep sorted order
        filename = f"{idx:05d}.png"
        cv2.imwrite(os.path.join(label_dir, filename), roi)
        burst_remaining -= 1
        if burst_remaining == 0:
            burst_label = None

    key = cv2.waitKey(10) & 0xFF
    if key in (ord('a'), ord('b'), ord('c')) and burst_label is None:
        label = chr(key).upper()
        label_dir = os.path.join(directory, label)
        burst_label = label
        burst_remaining = 30
        burst_start = next_index(label_dir)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()