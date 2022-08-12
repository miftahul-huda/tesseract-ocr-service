from flask import Flask
from flask import request
from flask import json
from flask.wrappers import Request
from ocr import draw_boxes, parse, parse_to_image, image_boxes, image_boxes_to_text, image_2dboxes_to_text
from six.moves import urllib
import os
app = Flask(__name__)

from flask import send_file


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ocr')
def ocr():
    url = request.args.get("url")
    #url = urllib.parse.unquote(url)
    print(url)
    boxes = parse(url)
    print(boxes)
    return json.jsonify(boxes)

@app.route('/ocr-to-image')
def ocr2image():
    url = request.args.get("url")
    #url = urllib.parse.unquote(url)
    print(url)
    filename = parse_to_image(url)
    return send_file(filename,  mimetype='image/png')

@app.route('/image-rect', methods=['GET', 'POST'])
def imagerect():
    url = request.args.get("url")
    content = request.json
    filename = image_boxes(url, content["positions"])
    #url = urllib.parse.unquote(url)
    print(url)
    return send_file(filename,  mimetype='image/png')

@app.route('/imageboxes2text', methods=['GET', 'POST'])
def imageboxes2text():
    url = request.args.get("url")
    content = request.json

    print("Positions")
    print(content["positions"])
    
    positions = image_boxes_to_text(url, content["positions"])
    return json.jsonify(positions)

@app.route('/image2dboxes2text', methods=['GET', 'POST'])
def image2dboxes2text():
    url = request.args.get("url")
    content = request.json

    print("Positions")
    print(content["positions"])
    
    positions = image_2dboxes_to_text(url, content["positions"])
    return json.jsonify(positions)

if __name__ == '__main__':
    app.run(port=os.environ['APPLICATION_PORT'])


