from turtle import position
import cv2
import pytesseract
from pytesseract import Output
from urllib.request import urlopen
from urllib.parse import quote
import numpy as np
import io
import uuid


def random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.


def parse(imageUrl):

    img = url_to_image(imageUrl)

    img = get_grayscale(img)
    # img = thresholding(img)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    n_boxes = len(d['text'])
    boxes = [];
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append({ 'x': x, 'y': y, 'w': w, 'h': h, 'text' : d['text'][i] })

    return boxes

def draw_boxes(imageUrl, boxes):
    img = url_to_image(imageUrl)

    print(boxes)
    for box in boxes:
        img = cv2.rectangle(img, (box['x'], box['y']), (box['x'] + box['width'], box['y'] + box['height']), (0, 255, 0), 2)
    
    filename = "/tmp/temp.png";
    cv2.imwrite(filename, img)

    return filename


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

def image_boxes(imageUrl, positions):

    img = url_to_image(imageUrl)

    print(positions)
    for position in positions:
        print ("position")
        print (position)
        img = cv2.rectangle(img, (int(position['x']), int(position['y'])), (int(position['x']) + int(position['w']), int(position['y']) + int(position['h'])), (0, 255, 0), 2)
        if 'text' in position.keys() :
            img = cv2.putText(img, position['text'], (int(position['x']) + 5, int(position['y']) + int(position['h']/2)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,0,0))
    
    filename = "/tmp/temp.png";
    cv2.imwrite(filename, img)

    return filename

def image_boxes_to_text(imageUrl, positions):

    img = url_to_image(imageUrl)
    #img = get_grayscale(img)
    #img = canny(img)

    for position in positions:

        croppedimg = img[int(position['y']):int(position['y']) + int(position['h']), int(position['x']):int(position['x']) + int(position['w'])]
        d = pytesseract.image_to_data(croppedimg, output_type=Output.DICT)

        text = ""
        l = len(d['text'])
        for i in range(l):
            if(len(d['text']) > 0):
                text = text + d['text'][i] + " "

        text = text.strip()
        position['text'] = text
        print ("position")
        print (position)
    

    return positions

def image_boxes_to_text_vision_api(imageUrl, positions):

    img = url_to_image(imageUrl)
    #img = get_grayscale(img)
    #img = canny(img)

    for position in positions:

        croppedimg = img[int(position['y']):int(position['y']) + int(position['h']), int(position['x']):int(position['x']) + int(position['w'])]
        #d = pytesseract.image_to_data(croppedimg, output_type=Output.DICT)

        filename = random_string(10)
        filename = "/tmp/" + filename + ".png";
        cv2.imwrite(filename, croppedimg)

        print("Detecting text in " + filename)
        text = detect_text(filename)
        text = text.strip()

        position['text'] = text
        print ("position")
        print (position)
    

    return positions

def image_2dboxes_to_text(imageUrl, rows):

    img = url_to_image(imageUrl)
    #img = remove_noise(img)
    #img = get_grayscale(img)
    #img = canny(img)

    newRows = []

    for row in rows:
        newRow = []
        total_empty = 0
        for position in row:

            croppedimg = img[int(position['y']):int(position['y']) + int(position['h']), int(position['x']):int(position['x']) + int(position['w'])]
            d = pytesseract.image_to_data(croppedimg, output_type=Output.DICT)

            text = ""
            l = len(d['text'])
            for i in range(l):
                if(len(d['text']) > 0):
                    text = text + d['text'][i] + " "

            text = text.strip()
            position['text'] = text
            print ("position")
            print (position)

            newRow.append(position)

            if len(text) == 0:
                total_empty = total_empty + 1

        if total_empty >= len(row):
            break
        else:
            newRows.append(newRow)

    #draw_image_2dboxes(imageUrl, rows)

    return newRows

def image_2dboxes_to_text_vision_api(imageUrl, rows):

    img = url_to_image(imageUrl)
    #img = remove_noise(img)
    #img = get_grayscale(img)
    #img = canny(img)

    newRows = []

    for row in rows:
        newRow = []
        total_empty = 0
        for position in row:

            croppedimg = img[int(position['y']):int(position['y']) + int(position['h']), int(position['x']):int(position['x']) + int(position['w'])]
            #d = pytesseract.image_to_data(croppedimg, output_type=Output.DICT)

            text = ""
            filename = random_string(10)
            filename = "/tmp/" + filename + ".png";
            cv2.imwrite(filename, croppedimg)

            print("Detecting text in " + filename)
            text = detect_text(filename)
            text = text.strip()
            position['text'] = text
            print ("position")
            print (position)

            newRow.append(position)

            if len(text) == 0:
                total_empty = total_empty + 1

        if total_empty >= len(row):
            break
        else:
            newRows.append(newRow)

    #draw_image_2dboxes(imageUrl, newRows)

    return newRows

def draw_image_2dboxes(imageUrl, rows):

    img = url_to_image(imageUrl)
    #img = get_grayscale(img)
    #img = canny(img)

    newRows = []

    for row in rows:
        newRow = []
        total_empty = 0
        for position in row:
            img2 = cv2.rectangle(img, (int(position['x']), int(position['y'])), (int(position['x']) + int(position['w']), int(position['y']) + int(position['h'])), (0, 255, 0), 2)
            if 'text' in position.keys() :
                img = cv2.putText(img, position['text'], (int(position['x']) + 5, int(position['y']) + int(position['h']/2)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,0,0))


    filename = "/tmp/temp.png";
    cv2.imwrite(filename, img)

    return filename



def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    #url = quote(url.encode("utf-8"), safe='')
    url = url.replace(' ', '%20')
    print("url")
    print(url)
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 




def detect_text(imagepath):
    """Detects text in the file located in Google Cloud Storage or on the Web.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(imagepath, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    #print(texts)

    idx  = 0
    concanetated_text  = ""
    for text in texts:
        print(text)
        if idx == 0:
            concanetated_text = concanetated_text + text.description + ' '
        idx = idx + 1

    if response.error.message:
        concanetated_text = ''

    return concanetated_text