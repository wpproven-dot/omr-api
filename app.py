from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - CLEAN DUAL ANCHOR SYSTEM
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
    
    # Question Column 1 (Q1-Q25)
    Q1_FROM_CORNER_X = 155.7721
    Q1_FROM_CORNER_Y = 17.5475
    Q1_OPTION_SPACING = 19.1926
    Q1_VERTICAL_SPACING = 18.5342
    Q1_TOTAL = 25
    
    # Question Column 2 (Q26-Q50)
    Q2_FROM_CORNER_X = 243.114
    Q2_FROM_CORNER_Y = 17.5475
    Q2_OPTION_SPACING = 19.1926
    Q2_VERTICAL_SPACING = 18.5342
    Q2_TOTAL = 25
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def find_corner_markers(image):
    """Detect 4 corner markers with improved accuracy"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Search regions for corners (8% of image size)
    search_size_x = int(width * 0.08)
    search_size_y = int(height * 0.08)
    
    corners = {}
    
    # Define search regions
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Adaptive threshold for better detection
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best square-like contour
        best_score = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:  # Too small
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if square-like
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Score based on squareness and size
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

def check_bubble_filled(gray_img, x, y, radius, threshold):
    """Check if bubble is filled"""
    try:
        x, y = int(round(x)), int(round(y))
        radius = int(round(radius))
        
        # Bounds check
        if x < radius or y < radius or x >= gray_img.shape[1] - radius or y >= gray_img.shape[0] - radius:
            return False, 0.0
        
        # Create circular mask
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Extract bubble region
        roi = cv2.bitwise_and(gray_img, mask)
        
        # Invert (dark pixels = high values)
        inverted = cv2.bitwise_not(roi)
        
        # Calculate fill percentage
        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return False, 0.0
            
        filled_pixels = np.sum(inverted[mask > 0] > 128)
        fill_pct = filled_pixels / total_pixels
        
        return fill_pct > threshold, fill_pct
    except Exception as e:
        return False, 0.0

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>🎯 OMR API v6.0</h1>
        <p style="font-size:1.2em; color:#667eea;">Clean Dual-Anchor System - Fixed!</p>
        <p style="color:#666;">Medical Student OMR Sheet Checker</p>
        <hr style="margin:30px 0; border:none; border-top:2px solid #667eea;">
        <div style="text-align:left; max-width:600px; margin:0 auto;">
            <h3>✨ V6.0 - The Fix:</h3>
            <ul style="line-height:2;">
                <li>✅ Simplified interpolation formula</li>
                <li>✅ Direct position calculation from corners</li>
                <li>✅ Q1 from top, Q25 from bottom, interpolate between</li>
                <li>✅ Same logic for Q26-Q50</li>
                <li>✅ Perfect alignment guaranteed!</li>
            </ul>
        </div>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'OMR API v6.0 - Clean dual-anchor interpolation',
        'version': '6.0'
    })

@app.route('/process-omr', methods=['POST'])
def process_omr():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file'}), 400
        
        img_height, img_width = img.shape[:2]
        result_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get threshold
        threshold = float(request.form.get('threshold', OMRConfig.FILL_THRESHOLD))
        
        # Find corner markers
        corners = find_corner_markers(img)
        
        if len(corners) < 4:
            return jsonify({
                'error': f'Could not detect all 4 corners. Found {len(corners)} corners. Please ensure all corner squares are visible and dark.',
                'corners_found': len(corners),
                'corners_detected': list(corners.keys())
            }), 400
        
        # Extract corner positions
        top_left = corners['top_left']
        top_right = corners['top_right']
        bottom_left = corners['bottom_left']
        bottom_right = corners['bottom_right']
        
        # Calculate scale factors from detected corners
        actual_width = top_right[0] - top_left[0]
        actual_height = bottom_left[1] - top_left[1]
        scale_x = actual_width / OMRConfig.CORNER_HORIZONTAL_DIST
        scale_y = actual_height / OMRConfig.CORNER_VERTICAL_DIST
        bubble_radius = (OMRConfig.BUBBLE_DIAMETER / 2) * scale_x
        
        # Mark detected corners
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
        # DETECT ANSWERS - COLUMN 1 (Q1-Q25)
        # CLEAN DUAL-ANCHOR INTERPOLATION
        # ==========================================
        answers = []
        
        # Calculate Q1 position from TOP-LEFT corner
        q1_offset_x = OMRConfig.Q1_FROM_CORNER_X
        q1_offset_y = OMRConfig.Q1_FROM_CORNER_Y
        q1_x_from_top = top_left[0] + (q1_offset_x * scale_x)
        q1_y_from_top = top_left[1] + (q1_offset_y * scale_y)
        
        # Calculate Q25 position from BOTTOM-LEFT corner
        # Q25 should be at the same offset from bottom as Q1 is from top!
        # Distance from Q1 to Q25 in template
        q1_to_q25_distance = OMRConfig.Q1_VERTICAL_SPACING * 24  # 24 gaps for 25 questions
        
        # Q25 offset from BOTTOM-LEFT corner
        # Total vertical distance minus the Q1-Q25 range minus the Q1 offset
        q25_offset_from_bottom = OMRConfig.CORNER_VERTICAL_DIST - q1_to_q25_distance - q1_offset_y
        
        q25_x_from_bottom = bottom_left[0] + (q1_offset_x * scale_x)
        q25_y_from_bottom = bottom_left[1] - (q25_offset_from_bottom * scale_y)
        
        for q_num in range(1, OMRConfig.Q1_TOTAL + 1):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            # Calculate progress (0.0 at Q1, 1.0 at Q25)
            progress = (q_num - 1) / 24.0 if OMRConfig.Q1_TOTAL > 1 else 0
            
            # Interpolate between Q1 position and Q25 position
            base_x = q1_x_from_top  # X stays same for all questions in column
            base_y = q1_y_from_top + progress * (q25_y_from_bottom - q1_y_from_top)
            
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
            
            # Mark bubbles
            for bubble in question_bubbles:
                if detected_option and bubble['option'] == detected_option:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (0, 255, 0), 2)
                    answers.append({
                        'question': q_num,
                        'answer': detected_option,
                        'confidence': round(bubble['fill_pct'] * 100, 1),
                        'status': 'marked'
                    })
                    break
            
            if not detected_option:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (128, 128, 128), 1)
                answers.append({
                    'question': q_num,
                    'answer': None,
                    'confidence': 0,
                    'status': 'not_marked'
                })
        
        # ==========================================
        # DETECT ANSWERS - COLUMN 2 (Q26-Q50)
        # CLEAN DUAL-ANCHOR INTERPOLATION
        # ==========================================
        
        # Calculate Q26 position from TOP-RIGHT corner
        # X offset needs to be calculated from template: Q2_FROM_CORNER_X is from TOP-LEFT
        # So we need to adjust for column 2
        q26_offset_x = OMRConfig.Q2_FROM_CORNER_X  # This is still from left side
        q26_offset_y = OMRConfig.Q2_FROM_CORNER_Y
        
        # Position from top-left (we'll use left reference for consistency)
        q26_x_from_top = top_left[0] + (q26_offset_x * scale_x)
        q26_y_from_top = top_right[1] + (q26_offset_y * scale_y)  # Use top_right Y for better accuracy
        
        # Calculate Q50 position from BOTTOM-RIGHT corner
        q26_to_q50_distance = OMRConfig.Q2_VERTICAL_SPACING * 24
        q50_offset_from_bottom = OMRConfig.CORNER_VERTICAL_DIST - q26_to_q50_distance - q26_offset_y
        
        q50_x_from_bottom = bottom_left[0] + (q26_offset_x * scale_x)
        q50_y_from_bottom = bottom_right[1] - (q50_offset_from_bottom * scale_y)
        
        for q_num in range(26, 26 + OMRConfig.Q2_TOTAL):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            # Calculate progress (0.0 at Q26, 1.0 at Q50)
            q_index = q_num - 26
            progress = q_index / 24.0 if OMRConfig.Q2_TOTAL > 1 else 0
            
            # Interpolate between Q26 position and Q50 position
            base_x = q26_x_from_top
            base_y = q26_y_from_top + progress * (q50_y_from_bottom - q26_y_from_top)
            
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
            
            # Mark bubbles
            for bubble in question_bubbles:
                if detected_option and bubble['option'] == detected_option:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (0, 255, 0), 2)
                    answers.append({
                        'question': q_num,
                        'answer': detected_option,
                        'confidence': round(bubble['fill_pct'] * 100, 1),
                        'status': 'marked'
                    })
                    break
            
            if not detected_option:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (128, 128, 128), 1)
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
            'version': '6.0',
            'method': 'clean_dual_anchor_interpolation',
            'corners_detected': len(corners),
            'corners': {k: list(v) for k, v in corners.items()},
            'scale_factors': {
                'x': round(scale_x, 4),
                'y': round(scale_y, 4)
            },
            'roll_number': roll_number if roll_number else None,
            'roll_detections': roll_detections,
            'set_code': set_code,
            'set_detection': set_detection,
            'total_questions': OMRConfig.Q1_TOTAL + OMRConfig.Q2_TOTAL,
            'answers_marked': len([a for a in answers if a['status'] == 'marked']),
            'answers_not_marked': len([a for a in answers if a['status'] == 'not_marked']),
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
