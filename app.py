from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import csv
from io import StringIO

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - LOOKUP TABLE ANCHOR SYSTEM + ANSWER KEY
# =====================================================
class OMRConfig:
    # Template dimensions
    TEMPLATE_WIDTH = 345.1207
    TEMPLATE_HEIGHT = 511.201
    
    # Corner square specifications
    CORNER_SQUARE_WIDTH = 9.951
    CORNER_SQUARE_HEIGHT = 9.6884
    
    # Verification distances
    CORNER_HORIZONTAL_DIST = 311.5073
    CORNER_VERTICAL_DIST = 466.7873
    
    # Bubble specifications
    BUBBLE_DIAMETER = 11.6013
    FILL_THRESHOLD = 0.35
    
    # Roll Number (6 digits, 0-9 vertical)
    ROLL_FROM_CORNER_X = 9.9484
    ROLL_FROM_CORNER_Y = 36.0871
    ROLL_VERTICAL_SPACING = 18.3243
    ROLL_HORIZONTAL_SPACING = 19.4086
    ROLL_DIGITS = 6
    ROLL_OPTIONS = 10
    
    # Set Code (A, B, C, D vertical)
    SET_FROM_CORNER_X = 126.4188
    SET_FROM_CORNER_Y = 36.0871
    SET_VERTICAL_SPACING = 18.3243
    SET_OPTIONS = ['A', 'B', 'C', 'D']
    
    # ANCHOR POINTS - Column 1 (measured from TOP-LEFT corner to Option A)
    COLUMN1_ANCHORS = {
        1:  {'x': 155.7764, 'y': 17.5495},
        5:  {'x': 155.7764, 'y': 90.3978},
        10: {'x': 155.7764, 'y': 181.7148},
        15: {'x': 155.7764, 'y': 272.8626},
        20: {'x': 155.7764, 'y': 364.1032},
        25: {'x': 155.7764, 'y': 455.0544}
    }
    
    # ANCHOR POINTS - Column 2 (measured from TOP-LEFT corner to Option A)
    COLUMN2_ANCHORS = {
        26: {'x': 243.1195, 'y': 17.5495},
        30: {'x': 243.1195, 'y': 90.3978},
        35: {'x': 243.1195, 'y': 181.7148},
        40: {'x': 243.1195, 'y': 272.8626},
        45: {'x': 243.1195, 'y': 364.1032},
        50: {'x': 243.1195, 'y': 455.0544}
    }
    
    # Option spacing (A→B→C→D going RIGHT)
    Q1_OPTION_SPACING = 19.1926
    Q2_OPTION_SPACING = 19.1926
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']
    Q1_TOTAL = 25
    Q2_TOTAL = 25

def find_corner_markers(image):
    """Detect 4 corner markers with improved accuracy"""
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
            if area < 30:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if 0.75 < aspect_ratio < 1.25:
                squareness = 1.0 - abs(1.0 - aspect_ratio)
                size_score = min(area / 200.0, 1.0)
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

def interpolate_position(anchors, question_num, scale_x, scale_y, top_left):
    """Get position for a question using piecewise linear interpolation"""
    anchor_questions = sorted(anchors.keys())
    
    if question_num in anchors:
        template_x = anchors[question_num]['x']
        template_y = anchors[question_num]['y']
        actual_x = top_left[0] + (template_x * scale_x)
        actual_y = top_left[1] + (template_y * scale_y)
        return (actual_x, actual_y)
    
    lower_anchor = None
    upper_anchor = None
    
    for anchor_q in anchor_questions:
        if anchor_q < question_num:
            lower_anchor = anchor_q
        elif anchor_q > question_num and upper_anchor is None:
            upper_anchor = anchor_q
            break
    
    if lower_anchor is None or upper_anchor is None:
        closest = min(anchor_questions, key=lambda x: abs(x - question_num))
        template_x = anchors[closest]['x']
        template_y = anchors[closest]['y']
        actual_x = top_left[0] + (template_x * scale_x)
        actual_y = top_left[1] + (template_y * scale_y)
        return (actual_x, actual_y)
    
    lower_data = anchors[lower_anchor]
    upper_data = anchors[upper_anchor]
    
    progress = (question_num - lower_anchor) / (upper_anchor - lower_anchor)
    
    template_x = lower_data['x'] + progress * (upper_data['x'] - lower_data['x'])
    template_y = lower_data['y'] + progress * (upper_data['y'] - lower_data['y'])
    
    actual_x = top_left[0] + (template_x * scale_x)
    actual_y = top_left[1] + (template_y * scale_y)
    
    return (actual_x, actual_y)

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
    except Exception as e:
        return False, 0.0

def parse_answer_key(csv_content):
    """Parse CSV answer key"""
    answer_key = {}
    try:
        reader = csv.DictReader(StringIO(csv_content))
        for row in reader:
            q_num = int(row.get('question', row.get('Question', row.get('Q', 0))))
            answer = row.get('answer', row.get('Answer', row.get('A', ''))).strip().upper()
            if q_num > 0 and answer in ['A', 'B', 'C', 'D']:
                answer_key[q_num] = answer
    except Exception as e:
        pass
    return answer_key

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>🎓 OMR Answer Checker v7.1</h1>
        <p style="font-size:1.2em; color:#667eea;">OMR Scoring Machine - With Answer Key</p>
        <p style="color:#666;">Automated Medical Exam Evaluation System</p>
        <hr style="margin:30px 0; border:none; border-top:2px solid #667eea;">
        <div style="text-align:left; max-width:600px; margin:0 auto;">
            <h3>Features:</h3>
            <ul style="line-height:2;">
                <li>✅ Detects filled bubbles with anchor-based positioning</li>
                <li>✅ Compares with answer key (CSV)</li>
                <li>✅ Green circle = Correct answer</li>
                <li>✅ Red circle = Wrong answer</li>
                <li>✅ Yellow circle = Skipped</li>
                <li>✅ Automatic scoring</li>
            </ul>
        </div>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'OMR Answer Checker v7.1',
        'version': '7.1'
    })

@app.route('/process-omr-with-key', methods=['POST'])
def process_omr_with_key():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No OMR image uploaded'}), 400
        
        if 'answer_key' not in request.files:
            return jsonify({'error': 'No answer key CSV uploaded'}), 400
        
        omr_file = request.files['file']
        key_file = request.files['answer_key']
        
        # Read CSV answer key
        answer_key_content = key_file.read().decode('utf-8')
        answer_key = parse_answer_key(answer_key_content)
        
        if not answer_key:
            return jsonify({'error': 'Invalid CSV format. Use columns: question, answer'}), 400
        
        # Save and read OMR image
        filepath = f'/tmp/{omr_file.filename}'
        omr_file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        img_height, img_width = img.shape[:2]
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        threshold = float(request.form.get('threshold', OMRConfig.FILL_THRESHOLD))
        
        # Find corner markers
        corners = find_corner_markers(img)
        
        if len(corners) < 4:
            return jsonify({
                'error': f'Could not detect all 4 corners. Found {len(corners)} corners.',
                'corners_found': len(corners)
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
            cv2.circle(result_img, (cx, cy), 7, (255, 0, 255), -1)
            cv2.putText(result_img, corner_name[:2].upper(), (cx + 12, cy - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # ==========================================
        # DETECT ROLL NUMBER
        # ==========================================
        roll_number = ""
        roll_detections = []
        
        for digit_col in range(OMRConfig.ROLL_DIGITS):
            detected_digit = None
            max_fill = 0
            best_detection = None
            
            for row in range(OMRConfig.ROLL_OPTIONS):
                offset_x = OMRConfig.ROLL_FROM_CORNER_X + (digit_col * OMRConfig.ROLL_HORIZONTAL_SPACING)
                offset_y = OMRConfig.ROLL_FROM_CORNER_Y + (row * OMRConfig.ROLL_VERTICAL_SPACING)
                
                actual_x = top_left[0] + (offset_x * scale_x)
                actual_y = top_left[1] + (offset_y * scale_y)
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
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
                          int(bubble_radius), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT SET CODE
        # ==========================================
        set_code = None
        set_detection = None
        max_fill = 0
        
        for idx, option in enumerate(OMRConfig.SET_OPTIONS):
            offset_x = OMRConfig.SET_FROM_CORNER_X
            offset_y = OMRConfig.SET_FROM_CORNER_Y + (idx * OMRConfig.SET_VERTICAL_SPACING)
            
            actual_x = top_left[0] + (offset_x * scale_x)
            actual_y = top_left[1] + (offset_y * scale_y)
            
            is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
            
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
                      int(bubble_radius), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT ANSWERS & COMPARE WITH KEY
        # ==========================================
        answers = []
        score_data = {
            'correct': 0,
            'incorrect': 0,
            'skipped': 0,
            'score_percentage': 0
        }
        
        # Process Column 1 (Q1-Q25)
        for q_num in range(1, OMRConfig.Q1_TOTAL + 1):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            base_x, base_y = interpolate_position(
                OMRConfig.COLUMN1_ANCHORS, 
                q_num, 
                scale_x, 
                scale_y, 
                top_left
            )
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                actual_x = base_x + (opt_idx * OMRConfig.Q1_OPTION_SPACING * scale_x)
                actual_y = base_y
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
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
            
            # Determine result
            correct_answer = answer_key.get(q_num, None)
            result_status = 'skipped'
            mark_color = (0, 255, 255)  # Yellow for skipped
            
            if detected_option is None:
                result_status = 'skipped'
                mark_color = (0, 255, 255)  # Yellow
                score_data['skipped'] += 1
            elif detected_option == correct_answer:
                result_status = 'correct'
                mark_color = (0, 255, 0)  # Green
                score_data['correct'] += 1
            else:
                result_status = 'incorrect'
                mark_color = (0, 0, 255)  # Red
                score_data['incorrect'] += 1
            
            # Mark bubbles
            for bubble in question_bubbles:
                if detected_option and bubble['option'] == detected_option:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), mark_color, 2)
                    answers.append({
                        'question': q_num,
                        'student_answer': detected_option,
                        'correct_answer': correct_answer,
                        'status': result_status,
                        'confidence': round(bubble['fill_pct'] * 100, 1)
                    })
                    break
            
            if not detected_option:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), mark_color, 1)
                answers.append({
                    'question': q_num,
                    'student_answer': None,
                    'correct_answer': correct_answer,
                    'status': result_status,
                    'confidence': 0
                })
        
        # Process Column 2 (Q26-Q50)
        for q_num in range(26, 26 + OMRConfig.Q2_TOTAL):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            base_x, base_y = interpolate_position(
                OMRConfig.COLUMN2_ANCHORS, 
                q_num, 
                scale_x, 
                scale_y, 
                top_left
            )
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                actual_x = base_x + (opt_idx * OMRConfig.Q2_OPTION_SPACING * scale_x)
                actual_y = base_y
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
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
            
            # Determine result
            correct_answer = answer_key.get(q_num, None)
            result_status = 'skipped'
            mark_color = (0, 255, 255)  # Yellow for skipped
            
            if detected_option is None:
                result_status = 'skipped'
                mark_color = (0, 255, 255)  # Yellow
                score_data['skipped'] += 1
            elif detected_option == correct_answer:
                result_status = 'correct'
                mark_color = (0, 255, 0)  # Green
                score_data['correct'] += 1
            else:
                result_status = 'incorrect'
                mark_color = (0, 0, 255)  # Red
                score_data['incorrect'] += 1
            
            # Mark bubbles
            for bubble in question_bubbles:
                if detected_option and bubble['option'] == detected_option:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), mark_color, 2)
                    answers.append({
                        'question': q_num,
                        'student_answer': detected_option,
                        'correct_answer': correct_answer,
                        'status': result_status,
                        'confidence': round(bubble['fill_pct'] * 100, 1)
                    })
                    break
            
            if not detected_option:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), mark_color, 1)
                answers.append({
                    'question': q_num,
                    'student_answer': None,
                    'correct_answer': correct_answer,
                    'status': result_status,
                    'confidence': 0
                })
        
        # Calculate score percentage
        total_questions = score_data['correct'] + score_data['incorrect'] + score_data['skipped']
        if total_questions > 0:
            score_data['score_percentage'] = round((score_data['correct'] / total_questions) * 100, 2)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'version': '7.1',
            'roll_number': roll_number if roll_number else None,
            'set_code': set_code,
            'scoring': score_data,
            'answers': answers,
            'threshold_used': threshold,
            'result_image': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e), 
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
