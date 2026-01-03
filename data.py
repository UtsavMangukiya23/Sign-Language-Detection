from function import *
import cv2
import os
import math

# Ensure target dataset structure exists
for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

def process_with_mediapipe():
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:
        for action in actions:
            image_dir = os.path.join('Image', action)
            if not os.path.isdir(image_dir):
                print(f'Warning: missing folder {image_dir}')
                continue

            image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
            if not image_files:
                print(f'No images found in {image_dir}')
                continue

            # Use ceil to cover all images; pad last sequence if needed
            num_sequences_available = math.ceil(len(image_files) / sequence_length)
            for seq_idx in range(num_sequences_available):
                seq_files = image_files[seq_idx * sequence_length : (seq_idx + 1) * sequence_length]
                if len(seq_files) < sequence_length:
                    seq_files += [seq_files[-1]] * (sequence_length - len(seq_files))
                os.makedirs(os.path.join(DATA_PATH, action, str(seq_idx)), exist_ok=True)

                for frame_num, fname in enumerate(seq_files):
                    frame_path = os.path.join(image_dir, fname)
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f'Warning: cannot read {frame_path}')
                        continue
                    image, results = mediapipe_detection(frame, hands)
                    draw_styles_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(seq_idx), f'{frame_num}.npy')
                    np.save(npy_path, keypoints)
                print(f'Processed {action} sequence {seq_idx}')

def process_without_mediapipe():
    print('Mediapipe not available; generating zero keypoints for dataset.')
    for action in actions:
        image_dir = os.path.join('Image', action)
        if not os.path.isdir(image_dir):
            print(f'Warning: missing folder {image_dir}')
            image_files = []
        else:
            image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        if not image_files:
            print(f'No images found in {image_dir}, creating dummy sequence.')
        # Use ceil to cover all images; pad last sequence if needed
        num_sequences_available = 1 if not image_files else math.ceil(len(image_files) / sequence_length)
        for seq_idx in range(num_sequences_available):
            os.makedirs(os.path.join(DATA_PATH, action, str(seq_idx)), exist_ok=True)
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, action, str(seq_idx), f'{frame_num}.npy')
                np.save(npy_path, np.zeros(21 * 3))
            print(f'Processed {action} sequence {seq_idx}')

if mp_hands:
    process_with_mediapipe()
else:
    process_without_mediapipe()

print('Keypoint extraction complete.')
                