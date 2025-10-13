from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE CONFIGURATION - FIXED FOR 2 COLUMNS
# =====================================================
class OMRTemplate:
    # Paper dimensions
    PAPER_WIDTH = 345.12
    PAPER_HEIGHT = 511.20
    
    # Bubble specifications
    BUBBLE_WIDTH = 11.60
    BUBBLE_THRESHOLD = 0.35
    
    # Roll Number section (6 digits, 10 rows each: 0-9)
    ROLL_START_X = 27.19
    ROLL_START_Y = 58.77
    ROLL_VERTICAL_SPACING = 18.33
    ROLL_HORIZONTAL_SPACING = 19.36
    ROLL_DIGITS = 6
    ROLL_OPTIONS = 10
    
    # Set Code section (A, B, C, D) - HORIZONTAL LAYOUT
    SET_START_X = 143.68
    SET_START_Y = 58.77
    SET_HORIZONTAL_SPACING = 18.33  # Space between A, B, C, D (going RIGHT)
    SET_OPTIONS = ['A', 'B', 'C', 'D']
    
    # Questions section - TWO COLUMNS
    # Column 1: Questions 1-25
    Q1_START_X = 173.02
    Q1_START_Y = 40.18
    Q1_OPTION_SPACING = 19.19  # Space between A, B, C, D
    Q1_VERTICAL_SPACING = 18.53  # Space between questions
    
    # Column 2: Questions 26-50 (adjust these based on actual position)
    Q2_START_X = 252.0  # Estimated - adjust if needed
    Q2_START_Y = 40.18
    Q2_OPTION_SPACING = 19.19
    Q2_VERTICAL_SPACING = 18.53
    
    Q_TOTAL = 50
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def check_bubble_filled(img_gray, x, y, bubble_size, threshold):
    """Check if a bubble at (x, y) is filled beyond threshold"""
    try:
        half_size = int(bubble_size / 2)
        x_int, y_int = int(x), int(y)
        
        roi = img_gray[max(0, y_int-half_size):min(img_gray.shape[0], y_int+half_size),
                       max(0, x_int-half_size):min(img_gray.shape[1], x_int+half_size)]
        
        if roi.size == 0:
            return False, 0.0
        
        inverted = cv2.bitwise_not(roi)
        total_pixels = roi.shape[0] * roi.shape[1]
        dark_pixels = np.sum(inverted > 128)
        fill_percentage = dark_pixels / total_pixels if total_pixels > 0 else 0
        
        is_filled = fill_percentage > threshold
        return is_filled, fill_percentage
        
    except Exception as e:
        return False, 0.0

def scale_coordinates(template_coord, img_width, img_height):
    """Scale template coordinates to actual image size"""
    scale_x = img_width / OMRTemplate.PAPER_WIDTH
    scale_y = img_height / OMRTemplate.PAPER_HEIGHT
    return template_coord[0] * scale_x, template_coord[1] * scale_y

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>🎯 OMR API is Running!</h1>
        <p>Medical Student OMR Sheet Checker - v2.1</p>
        <p style="color:#666;">2-Column Layout Support</p>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'OMR API v2.1 working!'})

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
        
        img_height, img_width = img.shape[:2]
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        threshold = float(request.form.get('threshold', OMRTemplate.BUBBLE_THRESHOLD))
        bubble_size = OMRTemplate.BUBBLE_WIDTH * (img_width / OMRTemplate.PAPER_WIDTH)
        
        # ==========================================
        # DETECT ROLL NUMBER (6 digits)
        # ==========================================
        roll_number = ""
        roll_detections = []
        
        for digit_col in range(OMRTemplate.ROLL_DIGITS):
            detected_digit = None
            max_fill = 0
            best_detection = None
            
            for row in range(OMRTemplate.ROLL_OPTIONS):
                template_x = OMRTemplate.ROLL_START_X + (digit_col * OMRTemplate.ROLL_HORIZONTAL_SPACING)
                template_y = OMRTemplate.ROLL_START_Y + (row * OMRTemplate.ROLL_VERTICAL_SPACING)
                
                actual_x, actual_y = scale_coordinates((template_x, template_y), img_width, img_height)
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_size, threshold)
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_digit = str(row)
                    best_detection = {
                        'digit_position': digit_col + 1,
                        'value': detected_digit,
                        'x': int(actual_x),
                        'y': int(actual_y),
                        'confidence': round(fill_pct * 100, 1)
                    }
            
            if best_detection:
                roll_number += detected_digit
                roll_detections.append(best_detection)
                cv2.circle(result_img, (best_detection['x'], best_detection['y']), 
                          int(bubble_size/2 + 2), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT SET CODE (A, B, C, D) - HORIZONTAL
        # ==========================================
        set_code = None
        set_detection = None
        max_fill = 0
        
        for idx, option in enumerate(OMRTemplate.SET_OPTIONS):
            # Set code bubbles are HORIZONTAL (side by side)
            template_x = OMRTemplate.SET_START_X + (idx * OMRTemplate.SET_HORIZONTAL_SPACING)
            template_y = OMRTemplate.SET_START_Y
            
            actual_x, actual_y = scale_coordinates((template_x, template_y), img_width, img_height)
            is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_size, threshold)
            
            if is_filled and fill_pct > max_fill:
                max_fill = fill_pct
                set_code = option
                set_detection = {
                    'set': set_code,
                    'x': int(actual_x),
                    'y': int(actual_y),
                    'confidence': round(fill_pct * 100, 1)
                }
        
        if set_detection:
            cv2.circle(result_img, (set_detection['x'], set_detection['y']), 
                      int(bubble_size/2 + 2), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT ANSWERS - TWO COLUMNS
        # ==========================================
        answers = []
        
        # Column 1: Questions 1-25
        for q_num in range(1, 26):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            for opt_idx, option in enumerate(OMRTemplate.Q_OPTIONS):
                template_x = OMRTemplate.Q1_START_X + (opt_idx * OMRTemplate.Q1_OPTION_SPACING)
                template_y = OMRTemplate.Q1_START_Y + ((q_num - 1) * OMRTemplate.Q1_VERTICAL_SPACING)
                
                actual_x, actual_y = scale_coordinates((template_x, template_y), img_width, img_height)
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_size, threshold)
                
                question_bubbles.append({
                    'option': option,
                    'x': int(actual_x),
                    'y': int(actual_y),
                    'filled': is_filled,
                    'fill_pct': fill_pct
                })
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_option = option
            
            # Mark bubbles
            if detected_option:
                for bubble in question_bubbles:
                    if bubble['option'] == detected_option:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_size/2 + 2), (0, 255, 0), 2)
                        answers.append({
                            'question': q_num,
                            'answer': detected_option,
                            'confidence': round(bubble['fill_pct'] * 100, 1),
                            'status': 'marked'
                        })
                    else:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_size/2 + 1), (0, 0, 255), 1)
            else:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_size/2 + 1), (0, 0, 255), 1)
                answers.append({
                    'question': q_num,
                    'answer': None,
                    'confidence': 0,
                    'status': 'not_marked'
                })
        
        # Column 2: Questions 26-50
        for q_num in range(26, 51):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            # Use column 2 coordinates
            for opt_idx, option in enumerate(OMRTemplate.Q_OPTIONS):
                template_x = OMRTemplate.Q2_START_X + (opt_idx * OMRTemplate.Q2_OPTION_SPACING)
                template_y = OMRTemplate.Q2_START_Y + ((q_num - 26) * OMRTemplate.Q2_VERTICAL_SPACING)
                
                actual_x, actual_y = scale_coordinates((template_x, template_y), img_width, img_height)
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_size, threshold)
                
                question_bubbles.append({
                    'option': option,
                    'x': int(actual_x),
                    'y': int(actual_y),
                    'filled': is_filled,
                    'fill_pct': fill_pct
                })
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_option = option
            
            # Mark bubbles
            if detected_option:
                for bubble in question_bubbles:
                    if bubble['option'] == detected_option:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_size/2 + 2), (0, 255, 0), 2)
                        answers.append({
                            'question': q_num,
                            'answer': detected_option,
                            'confidence': round(bubble['fill_pct'] * 100, 1),
                            'status': 'marked'
                        })
                    else:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_size/2 + 1), (0, 0, 255), 1)
            else:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_size/2 + 1), (0, 0, 255), 1)
                answers.append({
                    'question': q_num,
                    'answer': None,
                    'confidence': 0,
                    'status': 'not_marked'
                })
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'roll_number': roll_number if roll_number else None,
            'roll_detections': roll_detections,
            'set_code': set_code,
            'set_detection': set_detection,
            'total_questions': OMRTemplate.Q_TOTAL,
            'answers_marked': len([a for a in answers if a['status'] == 'marked']),
            'answers_not_marked': len([a for a in answers if a['status'] == 'not_marked']),
            'answers': answers,
            'threshold_used': threshold,
            'result_image': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
