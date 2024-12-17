import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import tkinter as tk
import mss
from PIL import Image, ImageTk, ImageDraw

def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "tensorflow", "opencv-python", "numpy", "pillow", "scikit-learn", "mss"])

b = os.path.abspath(".")
i = os.path.join(b, "data", "temp")
a = os.path.join(b, "data", "annotations")
m = os.path.join(b, "models")
l = os.path.join(b, "logs", "fit")

os.makedirs(i, exist_ok=True)
os.makedirs(a, exist_ok=True)
os.makedirs(m, exist_ok=True)
os.makedirs(l, exist_ok=True)

def p_i(img_path, bbox):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    x_min, y_min, x_max, y_max = bbox
    img = img[y_min:y_max, x_min:x_max]
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def p_a(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    anns = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        anns.append((label, (x_min, y_min, x_max, y_max)))
    return anns

def lozf(i, a):
    imgs, labels = [], []
    print(f"Loading data from {i} and {a}")
    for xml_file in os.listdir(a):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(a, xml_file)
            img_file = xml_file.replace('.xml', '.png')
            img_path = os.path.join(i, img_file)
            anns = p_a(xml_path)
            for label, bbox in anns:
                if label != 'CAPTCHA':
                    img = p_i(img_path, bbox)
                    imgs.append(img)
                    labels.append(label)
    print(f"Loaded {len(imgs)} images")
    print(f"Available labels: {set(labels)}")
    return np.array(imgs), np.array(labels)

imgs, labels = lozf(i, a)

if len(imgs) == 0 or len(labels) == 0:
    sys.exit(1)

label_dict = {char: idx for idx, char in enumerate(sorted(set(labels)))}
labels = np.array([label_dict[label] for label in labels])
labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_dict))

X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=42)

def train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(label_dict), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    l_ = os.path.join(l, "training-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = tf.keras.callbacks.TensorBoard(l=l_, histogram_freq=1)

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[tensorboard_cb])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    model.save(os.path.join(m, 'captcha_model.h5'))

def gui():
    def start_train():
        subprocess.Popen([sys.executable, __file__, "--train"])

    def start_infer():
        subprocess.Popen([sys.executable, __file__, "--infer"])

    def open_tb():
        subprocess.Popen(["tensorboard", "--logdir", l])

    root = tk.Tk()
    root.title('Captcha')

    frame = tk.Frame(root)
    frame.pack(pady=20)

    train_btn = tk.Button(frame, text='Start Training', command=start_train, width=20)
    train_btn.pack(pady=10)

    infer_btn = tk.Button(frame, text='Start Real-Time Inference', command=start_infer, width=30)
    infer_btn.pack(pady=10)

    tb_btn = tk.Button(frame, text='Open TensorBoard', command=open_tb, width=20)
    tb_btn.pack(pady=10)

    root.mainloop()

def realtime():
    model = tf.keras.models.load_model(os.path.join(m, 'captcha_model.h5'))
    rev_label_dict = {v: k for k, v in label_dict.items()}

    root = tk.Tk()
    root.title("Real-Time Inference")
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    canvas = tk.Canvas(root, width=screen_w, height=screen_h)
    canvas.pack()

    sct = mss.mss()

    def cappred():
        sct_img = sct.grab(sct.monitors[0])
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img_gray = img.convert("L")
        img_resized = img_gray.resize((28, 28))
        img_norm = np.array(img_resized) / 255.0
        input_img = np.expand_dims(img_norm, axis=(0, -1))

        if np.sum(img_norm) > 0.1: 
            preds = model.predict(input_img)
            pred_class = np.argmax(preds)
            pred_label = rev_label_dict[pred_class]

            img_display = img.copy()
            draw = ImageDraw.Draw(img_display)
            draw.rectangle([(10, 10), (100, 100)], outline="red", width=2)
            draw.text((10, 110), pred_label, fill="red")

            img_tk = ImageTk.PhotoImage(img_display)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.img_tk = img_tk 

        root.update()
        root.after(100, cappred)

    cappred()
    root.mainloop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--infer', action='store_true', help='Run real-time inference')
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.infer:
        realtime()
    else:
        install_deps()
        gui()
