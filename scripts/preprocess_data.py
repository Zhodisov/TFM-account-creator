import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def prei(image_path, bbox):
    i = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    xmin, ymin, xmax, ymax = bbox
    i = i[ymin:ymax, xmin:xmax]
    i = cv2.resize(i, (28, 28))
    i = i / 255.0
    i = np.expand_dims(i, axis=-1)
    return i

def pari(xml_file):
    t = ET.parse(xml_file)
    r = t.getroot()
    annotations = []
    for obj in r.findall('object'):
        l = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append((l, (xmin, ymin, xmax, ymax)))
    return annotations

def load(idir, adir):
    images = []
    labels = []
    for xml_file in os.listdir(adir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(adir, xml_file)
            i_f = xml_file.replace('.xml', '.png')
            image_path = os.path.join(idir, i_f)
            annotations = pari(xml_path)
            for label, bbox in annotations:
                if label != 'CAPTCHA':
                    image = prei(image_path, bbox)
                    images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)

idir = '../data/temp'
adir = '../data/annotations'

images, labels = load(idir, adir)

label_dict = {char: idx for idx, char in enumerate(sorted(set(labels)))}
labels = np.array([label_dict[label] for label in labels])
labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_dict))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

np.save('../data/X_train.npy', X_train)
np.save('../data/X_test.npy', X_test)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)
np.save('../data/label_dict.npy', label_dict)
