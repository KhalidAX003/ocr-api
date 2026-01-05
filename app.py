from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import easyocr
import pytesseract
import uuid

app = Flask(__name__)

# কনফিগ
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# EasyOCR লোড
reader = easyocr.Reader(['en'], gpu=False)
tess_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_load_image(img_path):
    """ইমেজ সেফলি লোড করে cv2 ফরম্যাটে"""
    try:
        img_pil = Image.open(img_path)
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"[Load Error] {e}")
        return None

def clean_text(text):
    """শুধুমাত্র অক্ষর ও সংখ্যা রাখে"""
    if not text:
        return ""
    cleaned = ''.join(c for c in text if c.isalnum())
    return cleaned

def ocr_easyocr(img_path):
    try:
        result = reader.readtext(img_path, detail=0)
        if result and len(result) > 0:
            text = ''.join(result).strip()
            return clean_text(text)
    except Exception as e:
        print(f"[EasyOCR Error] {e}")
    return None

def ocr_tesseract_basic(img_path):
    try:
        img = safe_load_image(img_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, config=tess_config)
        return clean_text(text.strip())
    except Exception as e:
        print(f"[Tesseract Basic Error] {e}")
    return None

def ocr_tesseract_advanced(img_path):
    try:
        img = safe_load_image(img_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # লাইন ডিটেক্ট এবং মুছে ফেলা
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=5)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(thresh, (x1,y1), (x2,y2), (255,255,255), 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        text = pytesseract.image_to_string(morph, config=tess_config)
        return clean_text(text.strip())
    except Exception as e:
        print(f"[Tesseract Advanced Error] {e}")
    return None

@app.route('/ocr', methods=['POST'])
def ocr_api():
    # ফাইল চেক
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided in request'
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No image file selected'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Only JPG, PNG, JPEG allowed.'
        }), 400

    # ফাইল সেভ করুন
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # OCR চালান
        result = ocr_easyocr(filepath)
        if result and len(result) >= 3:
            response = {
                'success': True,
                'method': 'EasyOCR',
                'text': result,
                'length': len(result)
            }
        else:
            result = ocr_tesseract_basic(filepath)
            if result and len(result) >= 3:
                response = {
                    'success': True,
                    'method': 'Tesseract Basic',
                    'text': result,
                    'length': len(result)
                }
            else:
                result = ocr_tesseract_advanced(filepath)
                if result and len(result) >= 3:
                    response = {
                        'success': True,
                        'method': 'Tesseract Advanced',
                        'text': result,
                        'length': len(result)
                    }
                else:
                    response = {
                        'success': False,
                        'error': 'Could not extract text from image',
                        'text': '',
                        'method': 'All methods failed'
                    }

    except Exception as e:
        response = {
            'success': False,
            'error': f'OCR processing failed: {str(e)}',
            'text': ''
        }

    finally:
        # টেম্প ফাইল মুছে ফেলুন
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'OCR API',
        'supported_formats': ['jpg', 'jpeg', 'png']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
