from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - CORNER MARKER BASED v3.1
# =====================================================
class OMRConfig:
    TEMPLATE_WIDTH = 345.1207
    TEMPLATE_HEIGHT = 511.201
    
    CORNER_SQUARE_WIDTH = 9.951
    CORNER_SQUARE_HEIGHT = 9.6884
    CORNER_HORIZONTAL_DIST = 311.5073
    CORNER_VERTICAL_DIST = 466.7873
    
    BUBBLE_DIAMETER = 11.6013
    FILL_THRESHOLD = 0.35
    
    ROLL_FROM_CORNER_X = 9.9484
    ROLL_FROM_CORNER_Y = 36.0871
    ROLL_VERTICAL_SPACING = 18.3243
    ROLL_HORIZONTAL_SPACING = 19.4086
    ROLL_DIGITS = 6
    ROLL_OPTIONS = 10
    
    SET_FROM_CORNER_X = 126.4188
    SET_FROM_CORNER_Y = 36.0871
    SET_VERTICAL_SPACING = 18.3243
    SET_OPTIONS = ['A', 'B', 'C', 'D']
    
    Q1_FROM_CORNER_X = 155.7721
    Q1_FROM_CORNER_Y = 17.5475
    Q1_OPTION_SPACING = 19.1926
    Q1_VERTICAL_SPACING = 18.5342
    Q1_TOTAL = 25
    
    Q2_FROM_CORNER_X = 243.114
    Q2_FROM_CORNER_Y = 17.5475
    Q2_OPTION_SPACING = 19.1926
    Q2_VERTICAL_SPACING = 18.5342
    Q2_TOTAL = 25
    
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def find_corner_markers(image):
    """Detect 4 corner markers with improved square detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Expected corner square size (with tolerance)
    expected_size = (OMRConfig.CORNER_SQUARE_WIDTH + OMRConfig.CORNER_SQUARE_HEIGHT) / 2
    size_tolerance = 0.4  # 40% tolerance
    
    search_size_x = int(width * 0.15)  # Increased search area
    search_size_y = int(height * 0.15)
    
    corners = {}
    
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Threshold to find dark objects
        _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = -1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate metrics
            aspect_ratio = w / float(h) if h > 0 else 0
            avg_size = (w + h) / 2
            
            # Score based on:
            # 1. Square-like shape (aspect ratio near 1.0)
            # 2. Size matching expected corner square size
            # 3. Distance from edge (corners should be near edges)
            
            squareness_score = 1.0 - abs(1.0 - aspect_ratio)  # 1.0 = perfect square
            
            size_diff = abs(avg_size - expected_size)
            size_score = 1.0 - min(size_diff / expected_size, 1.0)
            
            # Distance from corner edge
            if corner_name == 'top_left':
                edge_dist = x + y
            elif corner_name == 'top_right':
                edge_dist = (roi.shape[1] - x - w) + y
            elif corner_name == 'bottom_left':
                edge_dist = x + (roi.shape[0] - y - h)
            else:  # bottom_right
                edge_dist = (roi.shape[1] - x - w) + (roi.shape[0] - y - h)
            
            edge_score = 1.0 / (1.0 + edge_dist / 20.0)  # Prefer closer to edge
            
            # Combined score
            total_score = (squareness_score * 0.4) + (size_score * 0.4) + (edge_score * 0.2)
            
            # Only consider if reasonably square and right size
            if squareness_score > 0.6 and size_score > (1.0 - size_tolerance):
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = contour
        
        if best_candidate is not None:
            M = cv2.moments(best_candidate)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                corners[corner_name] = (cx, cy)
    
    return corners

def apply_perspective_correction(image, corners):
    """Apply 4-point perspective transform to straighten OMR"""
    if len(corners) != 4:
        return image, None
    
    # Source points (detected corners)
    src_points = np.float32([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_left'],
        corners['bottom_right']
    ])
    
    # Calculate dimensions based on corner distances
    width = int(OMRConfig.TEMPLATE_WIDTH)
    height = int(OMRConfig.TEMPLATE_HEIGHT)
    
    # Destination points (perfect rectangle)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    warped = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped, matrix

def scale_from_template(template_coord, scale_x, scale_y):
    """Scale template coordinates to actual image"""
    return (template_coord[0] * scale_x, template_coord[1] * scale_y)

def check_bubble_filled(gray_img, x, y, radius, threshold):
    """Check if bubble is filled"""
    try:
        x, y = int(x), int(y)
        radius = int(radius)
        
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
        filled_pixels = np.sum(inverted[mask > 0] > 128)
        fill_pct = filled_pixels / total_pixels if total_pixels > 0 else 0
        
        return fill_pct > threshold, fill_pct
    except:
        return False, 0.0

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1>🎯 OMR API v3.1</h1>
        <p style="font-size:1.2em; color:#667eea;">Perspective-Corrected Detection</p>
        <p style="color:#666;">Medical Student OMR Sheet Checker</p>
        <hr style="margin:30px 0; border:none; border-top:2px solid #667eea;">
        <div style="text-align:left; max-width:600px; margin:0 auto;">
            <h3>✨ Features:</h3>
            <ul style="line-height:2;">
                <li>✅ Improved corner detection (distinguishes squares from bubbles)</li>
                <li>✅ 4-point perspective correction (fixes skew & distortion)</li>
                <li>✅ Roll Number detection (6 digits)</li>
                <li>✅ Set Code detection (A/B/C/D)</li>
                <li>✅ 50 Questions (2 columns, pixel-perfect)</li>
            </ul>
        </div>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'OMR API v3.1 - Perspective corrected',
        'version': '3.1'
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
        
        original_img = img.copy()
        
        # Find corner markers
        corners = find_corner_markers(img)
        
        if len(corners) < 4:
            return jsonify({
                'error': f'Could not detect all 4 corners. Found: {list(corners.keys())}. Ensure corner squares are visible and dark.'
            }), 400
        
        # Apply perspective correction
        corrected_img, transform_matrix = apply_perspective_correction(img, corners)
        
        if corrected_img is None:
            return jsonify({'error': 'Failed to apply perspective correction'}), 400
        
        result_img = corrected_img.copy()
        gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        
        # After correction, image is now standardized to template size
        img_height, img_width = corrected_img.shape[:2]
        
        threshold = float(request.form.get('threshold', OMRConfig.FILL_THRESHOLD))
        
        # Calculate scale (should be close to 1.0 after perspective correction)
        scale_x = img_width / OMRConfig.TEMPLATE_WIDTH
        scale_y = img_height / OMRConfig.TEMPLATE_HEIGHT
        bubble_radius = (OMRConfig.BUBBLE_DIAMETER / 2) * scale_x
        
        # Corner reference (top-left is now at approximately 0,0)
        corner_x, corner_y = 0, 0
        
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
                
                scaled_offset = scale_from_template((offset_x, offset_y), scale_x, scale_y)
                actual_x = corner_x + scaled_offset[0]
                actual_y = corner_y + scaled_offset[1]
                
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
                          int(bubble_radius + 2), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT SET CODE
        # ==========================================
        set_code = None
        set_detection = None
        max_fill = 0
        
        for idx, option in enumerate(OMRConfig.SET_OPTIONS):
            offset_x = OMRConfig.SET_FROM_CORNER_X
            offset_y = OMRConfig.SET_FROM_CORNER_Y + (idx * OMRConfig.SET_VERTICAL_SPACING)
            
            scaled_offset = scale_from_template((offset_x, offset_y), scale_x, scale_y)
            actual_x = corner_x + scaled_offset[0]
            actual_y = corner_y + scaled_offset[1]
            
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
                      int(bubble_radius + 2), (0, 255, 0), 2)
        
        # ==========================================
        # DETECT ANSWERS - BOTH COLUMNS
        # ==========================================
        answers = []
        
        # Column 1 (Q1-Q25)
        for q_num in range(1, OMRConfig.Q1_TOTAL + 1):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                offset_x = OMRConfig.Q1_FROM_CORNER_X + (opt_idx * OMRConfig.Q1_OPTION_SPACING)
                offset_y = OMRConfig.Q1_FROM_CORNER_Y + ((q_num - 1) * OMRConfig.Q1_VERTICAL_SPACING)
                
                scaled_offset = scale_from_template((offset_x, offset_y), scale_x, scale_y)
                actual_x = corner_x + scaled_offset[0]
                actual_y = corner_y + scaled_offset[1]
                
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
            
            if detected_option:
                for bubble in question_bubbles:
                    if bubble['option'] == detected_option:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_radius + 2), (0, 255, 0), 2)
                        answers.append({
                            'question': q_num,
                            'answer': detected_option,
                            'confidence': round(bubble['fill_pct'] * 100, 1),
                            'status': 'marked'
                        })
                    else:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_radius), (0, 0, 255), 1)
            else:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (0, 0, 255), 1)
                answers.append({
                    'question': q_num,
                    'answer': None,
                    'confidence': 0,
                    'status': 'not_marked'
                })
        
        # Column 2 (Q26-Q50)
        for q_num in range(26, 26 + OMRConfig.Q2_TOTAL):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                offset_x = OMRConfig.Q2_FROM_CORNER_X + (opt_idx * OMRConfig.Q2_OPTION_SPACING)
                offset_y = OMRConfig.Q2_FROM_CORNER_Y + ((q_num - 26) * OMRConfig.Q2_VERTICAL_SPACING)
                
                scaled_offset = scale_from_template((offset_x, offset_y), scale_x, scale_y)
                actual_x = corner_x + scaled_offset[0]
                actual_y = corner_y + scaled_offset[1]
                
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
            
            if detected_option:
                for bubble in question_bubbles:
                    if bubble['option'] == detected_option:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_radius + 2), (0, 255, 0), 2)
                        answers.append({
                            'question': q_num,
                            'answer': detected_option,
                            'confidence': round(bubble['fill_pct'] * 100, 1),
                            'status': 'marked'
                        })
                    else:
                        cv2.circle(result_img, (bubble['x'], bubble['y']), 
                                  int(bubble_radius), (0, 0, 255), 1)
            else:
                for bubble in question_bubbles:
                    cv2.circle(result_img, (bubble['x'], bubble['y']), 
                              int(bubble_radius), (0, 0, 255), 1)
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
            'corners_detected': len(corners),
            'perspective_corrected': True,
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
