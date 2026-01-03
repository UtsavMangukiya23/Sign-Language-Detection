import cv2
import numpy as np
import os

# Try to import Mediapipe; fall back gracefully if unavailable
MEDIAPIPE_AVAILABLE = False
mp_drawing = None
mp_drawing_styles = None
mp_hands = None
try:
    import mediapipe as mp
    from mediapipe import solutions as mp_solutions
    mp_drawing = mp_solutions.drawing_utils
    mp_drawing_styles = mp_solutions.drawing_styles
    mp_hands = mp_solutions.hands
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

class _EmptyResults:
    multi_hand_landmarks = None

def mediapipe_detection(image, model):
    if not MEDIAPIPE_AVAILABLE or model is None:
        return image, _EmptyResults()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styles_landmarks(image, results):
    if not MEDIAPIPE_AVAILABLE:
        return
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmarks]).flatten()
        return rh
    return np.zeros(21 * 3)

DATA_PATH = os.path.join('MP_Data')
actions = ['A', 'B', 'C']
no_sequences = 30
sequence_length = 30