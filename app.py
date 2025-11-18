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
    
    # Roll Number (Supports 6 or 7 digits)
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
    CORNER_HORIZONTAL_DIST = 309.776
    CORNER_VERTICAL_DIST = 467.6418
    BUBBLE_DIAMETER = 11.0
    FILL_THRESHOLD = 0.35
    
    # Roll Number (Supports 6 or 7 digits)
    ROLL_FROM_CORNER_X = 13.2077
    ROLL_FROM_CORNER_Y = 41.1813
    ROLL_VERTICAL_SPACING = 16.6317
    ROLL_HORIZONTAL_SPACING = 17.442
    ROLL_DIGITS = 7
    ROLL_OPTIONS = 10
    
    # Serial Number (6 digits)
    SERIAL_FROM_CORNER_X = 13.0459
    SERIAL_FROM_CORNER_Y = 307.6707
    SERIAL_VERTICAL_SPACING = 16.6262
    SERIAL_HORIZONTAL_SPACING = 17.6616
    SERIAL_DIGITS = 6
    SERIAL_OPTIONS = 10
    
    # Questions (50 MCQ in 2 columns)
    Q1_FROM_CORNER_X = 152.6586
    Q1_FROM_CORNER_Y = 17.894
    Q1_OPTION_SPACING = 19.1976
    Q1_VERTICAL_SPACING = 18.2594
    Q1_TOTAL = 25
    
    Q2_FROM_CORNER_X = 239.9378
    Q2_FROM_CORNER_Y = 17.894
    Q2_OPTION_SPACING = 19.1976
    Q2_VERTICAL_SPACING = 18.2594
    Q2_TOTAL = 25
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def order_points(pts):
    """Order points in order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_corner_markers(image):
    """Detect 4 corner markers with improved accuracy"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Increased search area for better detection
    search_size_x = int(width * 0.12)
    search_size_y = int(height * 0.12)
    
    corners = {}
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Enhanced preprocessing for better corner detection
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or area > 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # More lenient aspect ratio for rotated squares
            if 0.6 < aspect_ratio < 1.4:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # Prefer 4-sided shapes
                squareness = 1.0 - abs(1.0 - aspect_ratio)
                size_score = min(area / 200.0, 1.0)
                shape_score = 1.0 if len(approx) == 4 else 0.7
                
                score = squareness * size_score * shape_score
                
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

def apply_perspective_correction(image, corners, config):
    """Apply perspective transformation to straighten the OMR sheet"""
    if len(corners) < 4:
        return None, None
    
    # Get corner points
    pts = np.array([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left']
    ], dtype="float32")
    
    # Order points properly
    rect = order_points(pts)
    
    # Calculate expected dimensions based on corner distance
    actual_width = config.CORNER_HORIZONTAL_DIST
    actual_height = config.CORNER_VERTICAL_DIST
    
    # Add padding for template edges (corners are not at exact edges)
    padding_x = config.CORNER_SQUARE_WIDTH * 2
    padding_y = config.CORNER_SQUARE_HEIGHT * 2
    
    output_width = int(actual_width + padding_x * 2)
    output_height = int(actual_height + padding_y * 2)
    
    # Destination points (straightened sheet)
    dst = np.array([
        [padding_x, padding_y],
        [actual_width + padding_x, padding_y],
        [actual_width + padding_x, actual_height + padding_y],
        [padding_x, actual_height + padding_y]
    ], dtype="float32")
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Apply transformation
    warped = cv2.warpPerspective(image, M, (output_width, output_height))
    
    # Return corrected image and new corner positions
    new_corners = {
        'top_left': (int(padding_x), int(padding_y)),
        'top_right': (int(actual_width + padding_x), int(padding_y)),
        'bottom_left': (int(padding_x), int(actual_height + padding_y)),
        'bottom_right': (int(actual_width + padding_x), int(actual_height + padding_y))
    }
    
    return warped, new_corners

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
    """Generic OMR processing with perspective correction"""
    
    # Step 1: Find corners in original image
    corners = find_corner_markers(img)
    if len(corners) < 4:
        return {'error': f'Could not detect all 4 corners. Found {len(corners)}'}
    
    # Step 2: Apply perspective correction to straighten the sheet
    corrected_img, new_corners = apply_perspective_correction(img, corners, config)
    
    if corrected_img is None:
        return {'error': 'Failed to apply perspective correction'}
    
    # Step 3: Process the straightened image
    result_img = corrected_img.copy()
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    
    # Use corrected corner positions
    top_left = new_corners['top_left']
    top_right = new_corners['top_right']
    bottom_left = new_corners['bottom_left']
    
    # Calculate scales (should be close to 1.0 now after correction)
    actual_width = top_right[0] - top_left[0]
    actual_height = bottom_left[1] - top_left[1]
    scale_x = actual_width / config.CORNER_HORIZONTAL_DIST
    scale_y = actual_height / config.CORNER_VERTICAL_DIST
    bubble_radius = (config.BUBBLE_DIAMETER / 2) * scale_x
    
    # Mark corners for verification
    for corner_name, (cx, cy) in new_corners.items():
        cv2.circle(result_img, (cx, cy), 6, (255, 0, 255), -1)
    
    # Detect Roll Number
    roll_number = ""
    detected_digits = []

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
            detected_digits.append({
                'digit': detected_digit,
                'position': digit_col,
                'x': best_x,
                'y': best_y
            })

    if len(detected_digits) >= 6:
        for item in detected_digits:
            roll_number += item['digit']
            cv2.circle(result_img, (item['x'], item['y']), int(bubble_radius), (0, 255, 0), 2)
    
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
    
    def process_column(start_q, total_q, base_x_offset, base_y_offset, v_spacing, opt_spacing):
        for q_num in range(start_q, start_q + total_q):
            detected_option = None
            max_fill = 0
            best_x, best_y = 0, 0
            all_bubble_positions = {}
            
            base_x = top_left[0] + (base_x_offset * scale_x)
            base_y = top_left[1] + (base_y_offset * scale_y) + ((q_num - start_q) * v_spacing * scale_y)
            
            for opt_idx, option in enumerate(config.Q_OPTIONS):
                actual_x = base_x + (opt_idx * opt_spacing * scale_x)
                actual_y = base_y
                all_bubble_positions[option] = (int(actual_x), int(actual_y))
                
                cv2.circle(result_img, (int(actual_x), int(actual_y)), int(bubble_radius), (200, 200, 200), 1)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                font_thickness = 2
                text_size = cv2.getTextSize(option, font, font_scale, font_thickness)[0]
                text_x = int(actual_x - text_size[0] / 2)
                text_y = int(actual_y + text_size[1] / 2)
                cv2.putText(result_img, option, (text_x, text_y), font, font_scale, (160, 160, 160), font_thickness, cv2.LINE_AA)
                
                is_filled, fill_pct = check_bubble_filled(gray, actual_x, actual_y, bubble_radius, threshold)
                
                if is_filled and fill_pct > max_fill:
                    max_fill = fill_pct
                    detected_option = option
                    best_x, best_y = int(actual_x), int(actual_y)
            
            correct_answer = answer_key.get(str(q_num))
            is_correct = False
            
            if detected_option:
                if correct_answer and detected_option == correct_answer:
                    cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 255, 0), -1)
                    is_correct = True
                else:
                    cv2.circle(result_img, (best_x, best_y), int(bubble_radius), (0, 0, 255), -1)
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
    if hasattr(config, 'Q4_FROM_CORNER_X'):
        process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, 
                      config.Q1_VERTICAL_SPACING, config.Q1_OPTION_SPACING)
        process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, 
                      config.Q2_VERTICAL_SPACING, config.Q2_OPTION_SPACING)
        process_column(51, config.Q3_TOTAL, config.Q3_FROM_CORNER_X, config.Q3_FROM_CORNER_Y, 
                      config.Q3_VERTICAL_SPACING, config.Q3_OPTION_SPACING)
        process_column(76, config.Q4_TOTAL, config.Q4_FROM_CORNER_X, config.Q4_FROM_CORNER_Y, 
                      config.Q4_VERTICAL_SPACING, config.Q4_OPTION_SPACING)
    else:
        process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, 
                      config.Q1_VERTICAL_SPACING, config.Q1_OPTION_SPACING)
        process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, 
                      config.Q2_VERTICAL_SPACING, config.Q2_OPTION_SPACING)
    
    # Add header with stats
    img_height, img_width = result_img.shape[:2]
    header_height = 100
    final_img = np.ones((img_height + header_height, img_width, 3), dtype=np.uint8) * 255
    
    final_img[header_height:, :] = result_img
    
    cv2.rectangle(final_img, (0, 0), (img_width - 1, img_height + header_height - 1), (0, 0, 0), 2)
    
    correct_count = len([a for a in answers if a['status'] == 'correct'])
    wrong_count = len([a for a in answers if a['status'] == 'wrong'])
    skipped_count = len([a for a in answers if a['status'] == 'skipped'])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_width = 180
    box_height = 60
    box_spacing = 20
    total_width = (box_width * 3) + (box_spacing * 2)
    x_start = (img_width - total_width) // 2
    y_top = 20
    
    def draw_rounded_rect_filled(img, pt1, pt2, color, radius=15):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    
    draw_rounded_rect_filled(final_img, (x_start, y_top), (x_start + box_width, y_top + box_height), (0, 128, 0), 15)
    cv2.putText(final_img, f'Correct:{correct_count}', (x_start + 20, y_top + 38), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    x_start += box_width + box_spacing
    draw_rounded_rect_filled(final_img, (x_start, y_top), (x_start + box_width, y_top + box_height), (0, 0, 200), 15)
    cv2.putText(final_img, f'Wrong:{wrong_count}', (x_start + 25, y_top + 38), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    x_start += box_width + box_spacing
    draw_rounded_rect_filled(final_img, (x_start, y_top), (x_start + box_width, y_top + box_height), (0, 0, 0), 15)
    cv2.putText(final_img, f'Skipped:{skipped_count}', (x_start + 15, y_top + 38), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
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
        <h1>OMR Checker API - Enhanced with Auto-Correction</h1>
        <p style="font-size:1.2em; color:#667eea;">Handles Rotated & Skewed Sheets (50 & 100 MCQ)</p>
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
   return jsonify({'status': 'ok', 'message': 'OMR Checker API - Ready with Auto-Correction!'})

@app.route('/process-omr', methods=['POST'])
def process_omr_auto():
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
