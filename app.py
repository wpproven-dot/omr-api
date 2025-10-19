from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - 100 MCQ (4 COLUMNS)
# =====================================================
class OMRConfig100:
    TEMPLATE_WIDTH = 434.5635
    TEMPLATE_HEIGHT = 682.7202
    CORNER_SQUARE_WIDTH = 8.3242
    CORNER_SQUARE_HEIGHT = 8.3242
    CORNER_HORIZONTAL_DIST = 385.9177
    CORNER_VERTICAL_DIST = 633.5445
    BUBBLE_DIAMETER = 11.0
    FILL_THRESHOLD = 0.35
    
    # Roll Number (7 digits)
    ROLL_FROM_CORNER_X = 15.1605
    ROLL_FROM_CORNER_Y = 40.2629
    ROLL_VERTICAL_SPACING = 16.6315
    ROLL_HORIZONTAL_SPACING = 17.4537
    ROLL_DIGITS = 7
    ROLL_OPTIONS = 10
    
    # Serial Number (6 digits)
    SERIAL_FROM_CORNER_X = 159.8187
    SERIAL_FROM_CORNER_Y = 37.3886
    SERIAL_VERTICAL_SPACING = 16.6266
    SERIAL_HORIZONTAL_SPACING = 17.6587
    SERIAL_DIGITS = 6
    SERIAL_OPTIONS = 10
    
    # Questions (100 MCQ in 4 columns)
    Q1_FROM_CORNER_X = 28.3268
    Q1_FROM_CORNER_Y = 247.6496
    Q1_OPTION_SPACING = 17.9692
    Q1_VERTICAL_SPACING = 15.4698
    Q1_TOTAL = 25
    
    Q2_FROM_CORNER_X = 124.3077
    Q2_FROM_CORNER_Y = 247.6496
    Q2_OPTION_SPACING = 17.9692
    Q2_VERTICAL_SPACING = 15.4698
    Q2_TOTAL = 25
    
    Q3_FROM_CORNER_X = 220.3095
    Q3_FROM_CORNER_Y = 247.6496
    Q3_OPTION_SPACING = 17.9692
    Q3_VERTICAL_SPACING = 15.4698
    Q3_TOTAL = 25
    
    Q4_FROM_CORNER_X = 318.3722
    Q4_FROM_CORNER_Y = 247.6496
    Q4_OPTION_SPACING = 17.9692
    Q4_VERTICAL_SPACING = 15.4698
    Q4_TOTAL = 25
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']

# =====================================================
# OMR TEMPLATE - 50 MCQ (2 COLUMNS)
# =====================================================
class OMRConfig50:
    TEMPLATE_WIDTH = 345.6
    TEMPLATE_HEIGHT = 511.2
    CORNER_SQUARE_WIDTH = 10.0
    CORNER_SQUARE_HEIGHT = 10.0
    CORNER_HORIZONTAL_DIST = 311.0764
    CORNER_VERTICAL_DIST = 468.2187
    BUBBLE_DIAMETER = 11.0
    FILL_THRESHOLD = 0.35
    
    # Roll Number (7 digits)
    ROLL_FROM_CORNER_X = 14.5028
    ROLL_FROM_CORNER_Y = 42.1802
    ROLL_VERTICAL_SPACING = 16.6352
    ROLL_HORIZONTAL_SPACING = 17.4359
    ROLL_DIGITS = 7
    ROLL_OPTIONS = 10
    
    # Serial Number (6 digits)
    SERIAL_FROM_CORNER_X = 22.9152
    SERIAL_FROM_CORNER_Y = 307.0002
    SERIAL_VERTICAL_SPACING = 16.6266
    SERIAL_HORIZONTAL_SPACING = 17.6559
    SERIAL_DIGITS = 6
    SERIAL_OPTIONS = 10
    
    # Questions (50 MCQ in 2 columns)
    Q1_FROM_CORNER_X = 153.952
    Q1_FROM_CORNER_Y = 18.8907
    Q1_OPTION_SPACING = 19.1976
    Q1_VERTICAL_SPACING = 18.2549
    Q1_TOTAL = 25
    
    Q2_FROM_CORNER_X = 241.2513
    Q2_FROM_CORNER_Y = 18.8907
    Q2_OPTION_SPACING = 19.1976
    Q2_VERTICAL_SPACING = 18.2549
    Q2_TOTAL = 25
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def find_corner_markers(image):
    """Detect 4 corner markers"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    search_size_x = int(width * 0.08)
    search_size_y = int(height * 0.08)
    
    corners = {}
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if 0.75 < aspect_ratio < 1.25:
                squareness = 1.0 - abs(1.0 - aspect_ratio)
                size_score = min(area / 150.0, 1.0)
                score = squareness * size_score
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                corners[corner_name] = (cx, cy)
    
    return corners

def check_bubble_filled(gray_img, x, y, radius, threshold):
    """Check if bubble is filled"""
    try:
        x, y = int(round(x)), int(round(y))
        radius = int(round(radius))
        
        if x < radius or y < radius or x >= gray_img.shape[1] - radius or y >= gray_img.shape[0] - radius:
            return False, 0.0
        
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(gray_img, mask)
        inverted = cv2.bitwise_not(roi)
        
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return False, 0.0
        
        filled_pixels = np.sum(inverted[mask > 0] > 128)
        fill_pct = filled_pixels / total_pixels
        
        return fill_pct > threshold, fill_pct
    except:
        return False, 0.0

def process_omr_sheet(img, config, threshold, answer_key):
    """Generic OMR processing for both 50 and 100 MCQ"""
    result_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find corners
    corners = find_corner_markers(img)
    if len(corners) < 4:
        return {'error': f'Could not detect all 4 corners. Found {len(corners)}'}
    
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']
    
    # Calculate scales
    actual_width = top_right[0] - top_left[0]
    actual_height = bottom_left[1] - top_left[1]
    scale_x = actual_width / config.CORNER_HORIZONTAL_DIST
    scale_y = actual_height / config.CORNER_VERTICAL_DIST
    bubble_radius = (config.BUBBLE_DIAMETER / 2) * scale_x
    
    # Mark corners
    for corner_name, (cx, cy) in corners.items():
        cv2.circle(result_img, (cx, cy), 6, (255, 0, 255), -1)
    
    # Detect Roll Number
    roll_number = ""
    for digit_col in range(config.ROLL_DIGITS):
        detected_digit = None
        max_fill = 0
        
        for row in range(config.ROLL_OPTIONS):
            offset_x = config.ROLL_FROM_CORNER_X + (digit_col * config.ROLL_HORIZONTAL_SPACING)
            offset_y = config.ROLL_FROM_CORNER_Y + (row * config.ROLL_VERTICAL_SPACING)
            actual_x = top_left[0] + (offset_x * scale_x)
            actual_y = top_left[1] + (offset_y * scale_y)
            
            is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
            
            if is_filled and fill_pct > max_fill:
                max_fill = fill_pct
                detected_digit = str(row)
                best_x, best_y = int(actual_x), int(actual_y)
        
        if detected_digit:
            roll_number += detected_digit
            cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 2)
    
    # Detect Serial Number
    serial_number = ""
    for digit_col in range(config.SERIAL_DIGITS):
        detected_digit = None
        max_fill = 0
        
        for row in range(config.SERIAL_OPTIONS):
            offset_x = config.SERIAL_FROM_CORNER_X + (digit_col * config.SERIAL_HORIZONTAL_SPACING)
            offset_y = config.SERIAL_FROM_CORNER_Y + (row * config.SERIAL_VERTICAL_SPACING)
            actual_x = top_left[0] + (offset_x * scale_x)
            actual_y = top_left[1] + (offset_y * scale_y)
            
            is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
            
            if is_filled and fill_pct > max_fill:
                max_fill = fill_pct
                detected_digit = str(row)
                best_x, best_y = int(actual_x), int(actual_y)
        
        if detected_digit:
            serial_number += detected_digit
            cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 2)
    
    # Detect Answers
    answers = []
    
    def process_column(start_q, total_q, base_x_offset, base_y_offset, v_spacing):
        for q_num in range(start_q, start_q + total_q):
            detected_option = None
            max_fill = 0
            best_x, best_y = 0, 0
            all_bubble_positions = {}
            
            base_x = top_left[0] + (base_x_offset * scale_x)
            base_y = top_left[1] + (base_y_offset * scale_y) + ((q_num - start_q) * v_spacing * scale_y)
            
            for opt_idx, option in enumerate(config.Q_OPTIONS):
                actual_x = base_x + (opt_idx * config.Q1_OPTION_SPACING * scale_x)
                actual_y = base_y
                all_bubble_positions[option] = (int(actual_x), int(actual_y))
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_option = option
                    best_x, best_y = int(actual_x), int(actual_y)
            
            correct_answer = answer_key.get(str(q_num))
            is_correct = False
            
            if detected_option:
                if correct_answer and detected_option == correct_answer:
                    cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 3)
                    is_correct = True
                else:
                    cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 0, 255), 3)
                    if correct_answer and correct_answer in all_bubble_positions:
                        correct_x, correct_y = all_bubble_positions[correct_answer]
                        cv2.circle(result_img, (correct_x, correct_y), int(bubble_radius), (0, 255, 0), 2)
            else:
                if correct_answer and correct_answer in all_bubble_positions:
                    correct_x, correct_y = all_bubble_positions[correct_answer]
                    cv2.circle(result_img, (correct_x, correct_y), int(bubble_radius), (0, 255, 0), 2)
            
            answers.append({
                'question': q_num,
                'answer': detected_option,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'confidence': round(max_fill * 100, 1) if detected_option else 0,
                'status': 'correct' if is_correct else ('wrong' if detected_option else 'skipped')
            })
    
    # Process columns based on config
    if hasattr(config, 'Q4_FROM_CORNER_X'):  # 100 MCQ
        process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, config.Q1_VERTICAL_SPACING)
        process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, config.Q2_VERTICAL_SPACING)
        process_column(51, config.Q3_TOTAL, config.Q3_FROM_CORNER_X, config.Q3_FROM_CORNER_Y, config.Q3_VERTICAL_SPACING)
        process_column(76, config.Q4_TOTAL, config.Q4_FROM_CORNER_X, config.Q4_FROM_CORNER_Y, config.Q4_VERTICAL_SPACING)
    else:  # 50 MCQ
        process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, config.Q1_VERTICAL_SPACING)
        process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, config.Q2_VERTICAL_SPACING)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    correct_count = len([a for a in answers if a['status'] == 'correct'])
    wrong_count = len([a for a in answers if a['status'] == 'wrong'])
    skipped_count = len([a for a in answers if a['status'] == 'skipped'])
    
    return {
        'success': True,
        'roll_number': roll_number if roll_number else None,
        'serial_number': serial_number if serial_number else None,
        'total_questions': len(answers),
        'correct': correct_count,
        'wrong': wrong_count,
        'skipped': skipped_count,
        'marked': correct_count + wrong_count,
        'answers': answers,
        'threshold_used': threshold,
        'result_image': f'data:image/jpeg;base64,{img_base64}'
    }

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>OMR Checker API - Dual Format Support</h1>
        <p style="font-size:1.2em; color:#667eea;">50 MCQ & 100 MCQ Support</p>
        <hr style="margin:30px 0;">
        <h3>Endpoints:</h3>
        <ul style="line-height:2;">
            <li><code>/process-omr</code> - Auto-detect format</li>
            <li><code>/process-omr-50</code> - Force 50 MCQ</li>
            <li><code>/process-omr-100</code> - Force 100 MCQ</li>
        </ul>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'OMR Checker API - Ready!'})

@app.route('/process-omr', methods=['POST'])
def process_omr_auto():
    """Auto-detect format or use mcq_format parameter"""
    try:
        mcq_format = request.form.get('mcq_format', '100')
        
        if mcq_format == '50':
            return process_omr_50()
        else:
            return process_omr_100()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-omr-50', methods=['POST'])
def process_omr_50():
    """Process 50 MCQ OMR"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        threshold = float(request.form.get('threshold', OMRConfig50.FILL_THRESHOLD))
        answer_key_json = request.form.get('answer_key', '{}')
        
        import json
        answer_key = json.loads(answer_key_json)
        
        result = process_omr_sheet(img, OMRConfig50, threshold, answer_key)
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-omr-100', methods=['POST'])
def process_omr_100():
    """Process 100 MCQ OMR"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        threshold = float(request.form.get('threshold', OMRConfig100.FILL_THRESHOLD))
        answer_key_json = request.form.get('answer_key', '{}')
        
        import json
        answer_key = json.loads(answer_key_json)
        
        result = process_omr_sheet(img, OMRConfig100, threshold, answer_key)
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
