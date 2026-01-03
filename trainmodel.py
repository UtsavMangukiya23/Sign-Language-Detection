from function import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for seq_idx in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(seq_idx), f'{frame_num}.npy')
            if os.path.exists(npy_path):
                res = np.load(npy_path)
                window.append(res)
            else:
                window.append(np.zeros(21 * 3))
        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

if len(x) == 0:
    raise RuntimeError('No training data found in MP_Data. Run data.py after collecting images.')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
logs_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=logs_dir)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 21 * 3)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[tb_callback], validation_data=(x_test, y_test))
model.summary()

model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save('model.h5')
print('Model saved successfully.')
