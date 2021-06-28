# tesseract-ocr-service
OCR service for images. Parse text from image and return bounding boxes. Using python and tesseract.


# API

Call the api to parse table from image and return the bounding boxes and its text:
http://<host>/ocr?url=<urltoimage>
  
Call the api to parse table from image and return the image with bounding boxes:
http://<host>/ocr-to-image?url=<urltoimage>  
  
# Installation
1. Install the requirements:
  pip install -r requirements.txt
  
2. Run: python app.py
