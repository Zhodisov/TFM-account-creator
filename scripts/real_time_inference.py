import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('../models/captcha_model.h5')
label_dict = np.load('../data/label_dict.npy', allow_pickle=True).item()
reverse_label_dict = {v: k for k, v in label_dict.items()}
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(gray, (28, 28))
    n = r / 255.0
    i = np.expand_dims(n, axis=(0, -1))
    p = model.predict(i)
    pc = np.argmax(p)
    pl = reverse_label_dict[pc]
    cv2.putText(frame, f'Prediction: {pl}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
