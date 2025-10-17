from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - 100 MCQ (4 COLUMNS) + 7-DIGIT ROLL + 6-DIGIT SERIAL
# =====================================================
class OMRConfig:
    # Template dimensions
    TEMPLATE_WIDTH = 434.5635
    TEMPLATE_HEIGHT = 682.7202
    
    # Corner square specifications
    CORNER_SQUARE_WIDTH = 8.3242
    CORNER_SQUARE_HEIGHT = 8.3242
    
    # Verification distances
    CORNER_HORIZONTAL_DIST = 385.9177
    CORNER_VERTICAL_DIST = 633.5445
    
    # Bubble specifications
    BUBBLE_DIAMETER = 11.0
    FILL_THRESHOLD = 0.35
    
    # Roll Number (7 digits, 0-9 vertical)
    ROLL_FROM_CORNER_X = 15.1605
    ROLL_FROM_CORNER_Y = 40.2629
    ROLL_VERTICAL_SPACING = 16.6315
    ROLL_HORIZONTAL_SPACING = 17.4537
    ROLL_DIGITS = 7
    ROLL_OPTIONS = 10
    
    # Serial Number (6 digits, 0-9 vertical)
    SERIAL_FROM_CORNER_X = 159.8187
    SERIAL_FROM_CORNER_Y = 37.3886
    SERIAL_VERTICAL_SPACING = 16.6266
    SERIAL_HORIZONTAL_SPACING = 17.6587
    SERIAL_DIGITS = 6
    SERIAL_OPTIONS = 10
    
    # Question Column 1 (Q1-Q25)
    Q1_FROM_CORNER_X = 28.3268
    Q1_FROM_CORNER_Y = 247.6496
    Q1_OPTION_SPACING = 17.9692
    Q1_VERTICAL_SPACING = 15.4698
    Q1_TOTAL = 25
    
    # Question Column 2 (Q26-Q50)
    Q2_FROM_CORNER_X = 124.3077
    Q2_FROM_CORNER_Y = 247.6496
    Q2_OPTION_SPACING = 17.9692
    Q2_VERTICAL_SPACING = 15.4698
    Q2_TOTAL = 25
    
    # Question Column 3 (Q51-Q75)
    Q3_FROM_CORNER_X = 220.3095
    Q3_FROM_CORNER_Y = 247.6496
    Q3_OPTION_SPACING = 17.9692
    Q3_VERTICAL_SPACING = 15.4698
    Q3_TOTAL = 25
    
    # Question Column 4 (Q76-Q100)
    Q4_FROM_CORNER_X = 318.3722
    Q4_FROM_CORNER_Y = 247.6496
    Q4_OPTION_SPACING = 17.9692
    Q4_VERTICAL_SPACING = 15.4698
    Q4_TOTAL = 25
    
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

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>OMR Checker - 100 MCQ (4 Columns)</h1>
        <p style="font-size:1.2em; color:#667eea;">7-Digit Roll + 6-Digit Serial + 100 Questions</p>
        <p style="color:#28a745; font-weight:bold;">✅ Green = Correct | ❌ Red = Wrong + Green shows correct answer</p>
        <hr style="margin:30px 0; border:none; border-top:2px solid #667eea;">
        <div style="text-align:left; max-width:600px; margin:0 auto;">
            <h3>Features:</h3>
            <ul style="line-height:2;">
                <li>Auto-detect 4 corner markers</li>
                <li>7-digit roll number detection</li>
                <li>6-digit serial number detection</li>
                <li>100 MCQ in 4 columns (25 each)</li>
                <li>Smart answer checking with color coding</li>
                <li>Green circle = Correct answer</li>
                <li>Red circle = Wrong (+ Green shows correct)</li>
            </ul>
        </div>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'OMR Checker API - 100 MCQ Format with Answer Checking'
    })

@app.route('/process-omr', methods=['POST'])
def process_omr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        threshold = float(request.form.get('threshold', OMRConfig.FILL_THRESHOLD))
        
        # Get answer key from request (sent by WordPress)
        answer_key_json = request.form.get('answer_key', '{}')
        import json
        answer_key = json.loads(answer_key_json)  # {1: 'A', 2: 'B', ...}
        
        # Find corner markers
        corners = find_corner_markers(img)
        
        if len(corners) < 4:
            return jsonify({
                'error': f'Could not detect all 4 corners. Found {len(corners)}'
            }), 400
        
        top_left = corners['top_left']
        top_right = corners['top_right']
        bottom_left = corners['bottom_left']
        bottom_right = corners['bottom_right']
        
        # Calculate scale factors
        actual_width = top_right[0] - top_left[0]
        actual_height = bottom_left[1] - top_left[1]
        scale_x = actual_width / OMRConfig.CORNER_HORIZONTAL_DIST
        scale_y = actual_height / OMRConfig.CORNER_VERTICAL_DIST
        bubble_radius = (OMRConfig.BUBBLE_DIAMETER / 2) * scale_x
        
        # Mark corners
        for corner_name, (cx, cy) in corners.items():
            cv2.circle(result_img, (cx, cy), 6, (255, 0, 255), -1)
        
        # ==========================================
        # DETECT ROLL NUMBER (7 digits)
        # ==========================================
        roll_number = ""
        roll_detections = []
        
        for digit_col in range(OMRConfig.ROLL_DIGITS):
            detected_digit = None
            max_fill = 0
            
            for row in range(OMRConfig.ROLL_OPTIONS):
                offset_x = OMRConfig.ROLL_FROM_CORNER_X + (digit_col * OMRConfig.ROLL_HORIZONTAL_SPACING)
                offset_y = OMRConfig.ROLL_FROM_CORNER_Y + (row * OMRConfig.ROLL_VERTICAL_SPACING)
                
                actual_x = top_left[0] + (offset_x * scale_x)
                actual_y = top_left[1] + (offset_y * scale_y)
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_digit = str(row)
                    best_x, best_y = int(actual_x), int(actual_y)
            
            if detected_digit:
                roll_number += detected_digit
                roll_detections.append({'digit': digit_col + 1, 'value': detected_digit, 'confidence': round(max_fill * 100, 1)})
                cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT SERIAL NUMBER (6 digits)
        # ==========================================
        serial_number = ""
        serial_detections = []
        
        for digit_col in range(OMRConfig.SERIAL_DIGITS):
            detected_digit = None
            max_fill = 0
            
            for row in range(OMRConfig.SERIAL_OPTIONS):
                offset_x = OMRConfig.SERIAL_FROM_CORNER_X + (digit_col * OMRConfig.SERIAL_HORIZONTAL_SPACING)
                offset_y = OMRConfig.SERIAL_FROM_CORNER_Y + (row * OMRConfig.SERIAL_VERTICAL_SPACING)
                
                actual_x = top_left[0] + (offset_x * scale_x)
                actual_y = top_left[1] + (offset_y * scale_y)
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_digit = str(row)
                    best_x, best_y = int(actual_x), int(actual_y)
            
            if detected_digit:
                serial_number += detected_digit
                serial_detections.append({'digit': digit_col + 1, 'value': detected_digit, 'confidence': round(max_fill * 100, 1)})
                cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT ANSWERS - ALL 4 COLUMNS (100 MCQ)
        # WITH COLOR CODING
        # ==========================================
        answers = []
        
        # Helper function to process each column
        def process_column(start_q, total_q, base_x_offset, base_y_offset, v_spacing):
            for q_num in range(start_q, start_q + total_q):
                detected_option = None
                max_fill = 0
                best_x, best_y = 0, 0
                all_bubble_positions = {}
                
                base_x = top_left[0] + (base_x_offset * scale_x)
                base_y = top_left[1] + (base_y_offset * scale_y) + ((q_num - start_q) * v_spacing * scale_y)
                
                # Check all options and store positions
                for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                    actual_x = base_x + (opt_idx * OMRConfig.Q1_OPTION_SPACING * scale_x)
                    actual_y = base_y
                    
                    all_bubble_positions[option] = (int(actual_x), int(actual_y))
                    
                    is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                    
                    if is_filled and fill_pct > max_fill:
                        max_fill = fill_pct
                        detected_option = option
                        best_x, best_y = int(actual_x), int(actual_y)
                
                # Determine if answer is correct
                correct_answer = answer_key.get(str(q_num))
                is_correct = False
                
                if detected_option:
                    if correct_answer and detected_option == correct_answer:
                        # CORRECT - Green circle
                        cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), 3)
                        is_correct = True
                    else:
                        # WRONG - Red circle on student answer
                        cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 0, 255), 3)
                        
                        # Show correct answer with green circle
                        if correct_answer and correct_answer in all_bubble_positions:
                            correct_x, correct_y = all_bubble_positions[correct_answer]
                            cv2.circle(result_img, (correct_x, correct_y), int(bubble_radius), (0, 255, 0), 2)
                else:
                    # SKIPPED - No answer detected, show correct answer with green circle
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
        
        # Process all 4 columns
        process_column(1, OMRConfig.Q1_TOTAL, OMRConfig.Q1_FROM_CORNER_X, OMRConfig.Q1_FROM_CORNER_Y, OMRConfig.Q1_VERTICAL_SPACING)
        process_column(26, OMRConfig.Q2_TOTAL, OMRConfig.Q2_FROM_CORNER_X, OMRConfig.Q2_FROM_CORNER_Y, OMRConfig.Q2_VERTICAL_SPACING)
        process_column(51, OMRConfig.Q3_TOTAL, OMRConfig.Q3_FROM_CORNER_X, OMRConfig.Q3_FROM_CORNER_Y, OMRConfig.Q3_VERTICAL_SPACING)
        process_column(76, OMRConfig.Q4_TOTAL, OMRConfig.Q4_FROM_CORNER_X, OMRConfig.Q4_FROM_CORNER_Y, OMRConfig.Q4_VERTICAL_SPACING)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        os.remove(filepath)
        
        # Calculate stats
        correct_count = len([a for a in answers if a['status'] == 'correct'])
        wrong_count = len([a for a in answers if a['status'] == 'wrong'])
        skipped_count = len([a for a in answers if a['status'] == 'skipped'])
        
        return jsonify({
            'success': True,
            'roll_number': roll_number if roll_number else None,
            'serial_number': serial_number if serial_number else None,
            'total_questions': 100,
            'correct': correct_count,
            'wrong': wrong_count,
            'skipped': skipped_count,
            'marked': correct_count + wrong_count,
            'answers': answers,
            'threshold_used': threshold,
            'result_image': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
