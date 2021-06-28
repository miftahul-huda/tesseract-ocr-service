import cv2
import pytesseract
from pytesseract import Output
from urllib.request import urlopen
import numpy as np


def parse(imageUrl):

    img = url_to_image(imageUrl)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    n_boxes = len(d['text'])
    boxes = [];
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append({ 'x': x, 'y': y, 'w': w, 'h': h, 'text' : d['text'][i] })

    return boxes

def parse_to_image(imageUrl):

    img = url_to_image(imageUrl)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    n_boxes = len(d['text'])
    boxes = [];
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append({ 'x': x, 'y': y, 'w': w, 'h': h })
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    filename = "/tmp/temp.png";
    cv2.imwrite(filename, img)

    return filename

def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image