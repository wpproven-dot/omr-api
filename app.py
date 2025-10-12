from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>🎯 OMR API is Running!</h1>
        <p>For medical students - OMR sheet checker</p>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'API working!'})

@app.route('/process-omr', methods=['POST'])
def process_omr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Save temporarily
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        # Simple bubble detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                if 0.75 < w/h < 1.25:
                    bubbles.append({'x': x, 'y': y, 'cy': y + h//2})
        
        # Group into rows
        bubbles.sort(key=lambda b: b['cy'])
        rows = []
        current = []
        last_y = -1000
        
        for b in bubbles:
            if abs(b['cy'] - last_y) < 35:
                current.append(b)
            else:
                if current:
                    rows.append(sorted(current, key=lambda x: x['x']))
                current = [b]
                last_y = b['cy']
        
        if current:
            rows.append(sorted(current, key=lambda x: x['x']))
        
        # Extract answers
        answers = []
        opts = ['A','B','C','D','E','F']
        
        for i, row in enumerate(rows, 1):
            if len(row) >= 2:
                answers.append({
                    'question': i,
                    'answer': opts[0] if len(row) > 0 else '?'
                })
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'total_bubbles': len(bubbles),
            'total_questions': len(rows),
            'answers': answers
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
