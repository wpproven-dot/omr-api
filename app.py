from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

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
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        # Read and process image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image'}), 400
        
        result_img = img.copy()
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours (bubbles)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 2000:
                (x, y, w, h) = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                
                if 0.75 < aspect_ratio < 1.25:
                    # Measure darkness (how filled the bubble is)
                    roi = thresh[y:y+h, x:x+w]
                    filled_pixels = cv2.countNonZero(roi)
                    total_pixels = w * h
                    fill_percentage = filled_pixels / total_pixels if total_pixels > 0 else 0
                    
                    bubbles.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2,
                        'fill': fill_percentage
                    })
        
        # Sort bubbles by Y position (top to bottom)
        bubbles.sort(key=lambda b: b['center_y'])
        
        # Group into rows (questions)
        rows = []
        current_row = []
        last_y = -1000
        
        for b in bubbles:
            if abs(b['center_y'] - last_y) < 35:
                current_row.append(b)
            else:
                if current_row:
                    rows.append(sorted(current_row, key=lambda x: x['center_x']))
                current_row = [b]
                last_y = b['center_y']
        
        if current_row:
            rows.append(sorted(current_row, key=lambda x: x['center_x']))
        
        # Extract answers (find darkest bubble in each row)
        answers = []
        options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        for q_num, row in enumerate(rows, 1):
            if len(row) >= 2:
                # Find the DARKEST (most filled) bubble
                darkest_idx = max(range(len(row)), key=lambda i: row[i]['fill'])
                darkest_fill = row[darkest_idx]['fill']
                
                # Only consider as marked if > 30% filled
                if darkest_fill > 0.30:
                    answer_letter = options[darkest_idx] if darkest_idx < len(options) else '?'
                    answers.append({
                        'question': q_num,
                        'answer': answer_letter,
                        'confidence': round(darkest_fill * 100, 1)
                    })
                    
                    # Draw GREEN circle on filled bubble
                    bubble = row[darkest_idx]
                    cv2.circle(result_img, 
                              (bubble['center_x'], bubble['center_y']), 
                              max(bubble['w'], bubble['h']) // 2 + 3, 
                              (0, 255, 0), 3)
                else:
                    # Draw RED circles on unfilled bubbles
                    for bubble in row:
                        cv2.circle(result_img, 
                                  (bubble['center_x'], bubble['center_y']), 
                                  max(bubble['w'], bubble['h']) // 2 + 3, 
                                  (0, 0, 255), 2)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'total_bubbles': len(bubbles),
            'total_questions': len(rows),
            'answers': answers,
            'result_image': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
