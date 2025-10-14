from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
CORS(app)

# =====================================================
# OMR TEMPLATE - ADAPTIVE CORNER DETECTION
# =====================================================
class OMRConfig:
    # Template dimensions (from your measurements)
    TEMPLATE_WIDTH = 345.1207
    TEMPLATE_HEIGHT = 511.201
    
    # Corner square specifications
    CORNER_SQUARE_WIDTH = 9.951
    CORNER_SQUARE_HEIGHT = 9.6884
    
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
    
    # Options per question
    Q_OPTIONS = ['A', 'B', 'C', 'D']

def find_corner_markers_adaptive(image):
    """Adaptive corner detection - tries multiple methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Method 1: Strict detection (good for high quality scans)
    corners = find_corners_strict(gray, width, height)
    if len(corners) == 4:
        return corners, "strict"
    
    # Method 2: Relaxed detection (for varied quality)
    corners = find_corners_relaxed(gray, width, height)
    if len(corners) == 4:
        return corners, "relaxed"
    
    # Method 3: Simple detection (fallback)
    corners = find_corners_simple(gray, width, height)
    return corners, "simple"

def find_corners_strict(gray, width, height):
    """Strict corner detection with tight filtering"""
    search_pct = 0.08
    search_size_x = int(width * search_pct)
    search_size_y = int(height * search_pct)
    
    corners = {}
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 800:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
            if len(approx) < 4 or len(approx) > 6:
                continue
            
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            
            if fill_ratio < 0.65:
                continue
            
            score = fill_ratio * (1.0 - abs(1.0 - aspect_ratio))
            
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

def find_corners_relaxed(gray, width, height):
    """Relaxed corner detection with looser filtering"""
    search_pct = 0.10
    search_size_x = int(width * search_pct)
    search_size_y = int(height * search_pct)
    
    corners = {}
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Binary threshold
        _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_score = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40 or area > 1000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if aspect_ratio < 0.6 or aspect_ratio > 1.5:
                continue
            
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            
            if fill_ratio < 0.5:
                continue
            
            score = area * fill_ratio
            
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

def find_corners_simple(gray, width, height):
    """Simple corner detection - finds darkest regions"""
    search_pct = 0.12
    search_size_x = int(width * search_pct)
    search_size_y = int(height * search_pct)
    
    corners = {}
    regions = {
        'top_left': (0, search_size_x, 0, search_size_y),
        'top_right': (width - search_size_x, width, 0, search_size_y),
        'bottom_left': (0, search_size_x, height - search_size_y, height),
        'bottom_right': (width - search_size_x, width, height - search_size_y, height)
    }
    
    for corner_name, (x1, x2, y1, y2) in regions.items():
        roi = gray[y1:y2, x1:x2]
        
        # Find darkest region
        _, thresh = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30 and area > max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Very loose aspect ratio check
                if 0.5 < aspect_ratio < 2.0:
                    max_area = area
                    best_contour = contour
        
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x1
                cy = int(M["m01"] / M["m00"]) + y1
                corners[corner_name] = (cx, cy)
    
    return corners

def perspective_transform(image, corners):
    """Apply perspective transformation to correct skew/rotation"""
    if len(corners) != 4:
        return None, None
    
    src_points = np.float32([
        corners['top_left'],
        corners['top_right'],
        corners['bottom_right'],
        corners['bottom_left']
    ])
    
    dst_width = int(OMRConfig.TEMPLATE_WIDTH)
    dst_height = int(OMRConfig.TEMPLATE_HEIGHT)
    
    dst_points = np.float32([
        [0, 0],
        [dst_width, 0],
        [dst_width, dst_height],
        [0, dst_height]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (dst_width, dst_height))
    
    return warped, matrix

def check_bubble_filled(gray_img, x, y, radius, threshold):
    """Check if bubble is filled"""
    try:
        x, y = int(round(x)), int(round(y))
        radius = int(round(radius))
        
        h, w = gray_img.shape
        if x < radius or y < radius or x >= w - radius or y >= h - radius:
            return False, 0.0
        
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        roi = cv2.bitwise_and(gray_img, mask)
        inverted = cv2.bitwise_not(roi)
        
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
        <h1>🎯 OMR API v4.1</h1>
        <p style="font-size:1.2em; color:#667eea;">Adaptive Corner Detection</p>
        <p style="color:#666;">Medical Student OMR Sheet Checker</p>
        <hr style="margin:30px 0; border:none; border-top:2px solid #667eea;">
        <div style="text-align:left; max-width:600px; margin:0 auto;">
            <h3>✨ Features:</h3>
            <ul style="line-height:2;">
                <li>✅ Adaptive corner detection (3 methods)</li>
                <li>✅ Perspective correction</li>
                <li>✅ Works with varied scan quality</li>
                <li>✅ Automatic fallback detection</li>
            </ul>
        </div>
    </div>
    '''

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'OMR API v4.1 - Adaptive detection',
        'version': '4.1'
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
        threshold = float(request.form.get('threshold', OMRConfig.FILL_THRESHOLD))
        
        # Adaptive corner detection
        corners, detection_method = find_corner_markers_adaptive(img)
        
        if len(corners) != 4:
            return jsonify({
                'error': f'Could not detect all 4 corner markers. Found {len(corners)}/4. Please ensure corner squares are visible and dark.',
                'corners_found': len(corners),
                'detected_corners': list(corners.keys()),
                'detection_method': detection_method
            }), 400
        
        # Perspective transformation
        warped, transform_matrix = perspective_transform(img, corners)
        
        if warped is None:
            return jsonify({'error': 'Failed to apply perspective correction'}), 400
        
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        result_img = warped.copy()
        
        bubble_radius = OMRConfig.BUBBLE_DIAMETER / 2
        
        # Mark corners on corrected image
        corrected_corners = {
            'top_left': (0, 0),
            'top_right': (int(OMRConfig.TEMPLATE_WIDTH), 0),
            'bottom_left': (0, int(OMRConfig.TEMPLATE_HEIGHT)),
            'bottom_right': (int(OMRConfig.TEMPLATE_WIDTH), int(OMRConfig.TEMPLATE_HEIGHT))
        }
        
        for corner_name, (cx, cy) in corrected_corners.items():
            cv2.circle(result_img, (cx, cy), 5, (255, 0, 255), -1)
        
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
                actual_x = OMRConfig.ROLL_FROM_CORNER_X + (digit_col * OMRConfig.ROLL_HORIZONTAL_SPACING)
                actual_y = OMRConfig.ROLL_FROM_CORNER_Y + (row * OMRConfig.ROLL_VERTICAL_SPACING)
                
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
            actual_x = OMRConfig.SET_FROM_CORNER_X
            actual_y = OMRConfig.SET_FROM_CORNER_Y + (idx * OMRConfig.SET_VERTICAL_SPACING)
            
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
        # DETECT ANSWERS - COLUMN 1 (Q1-Q25)
        # ==========================================
        answers = []
        
        for q_num in range(1, OMRConfig.Q1_TOTAL + 1):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                actual_x = OMRConfig.Q1_FROM_CORNER_X + (opt_idx * OMRConfig.Q1_OPTION_SPACING)
                actual_y = OMRConfig.Q1_FROM_CORNER_Y + ((q_num - 1) * OMRConfig.Q1_VERTICAL_SPACING)
                
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
        
        # ==========================================
        # DETECT ANSWERS - COLUMN 2 (Q26-Q50)
        # ==========================================
        for q_num in range(26, 26 + OMRConfig.Q2_TOTAL):
            detected_option = None
            max_fill = 0
            question_bubbles = []
            
            for opt_idx, option in enumerate(OMRConfig.Q_OPTIONS):
                actual_x = OMRConfig.Q2_FROM_CORNER_X + (opt_idx * OMRConfig.Q2_OPTION_SPACING)
                actual_y = OMRConfig.Q2_FROM_CORNER_Y + ((q_num - 26) * OMRConfig.Q2_VERTICAL_SPACING)
                
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
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Original with corners
        for corner_name, (cx, cy) in corners.items():
            cv2.circle(original_img, (cx, cy), 8, (255, 0, 255), -1)
            cv2.putText(original_img, corner_name[:2].upper(), (cx + 12, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        _, orig_buffer = cv2.imencode('.jpg', original_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        orig_base64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'detection_method': detection_method,
            'corners_detected': len(corners),
            'original_corners': {k: list(v) for k, v in corners.items()},
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
            'result_image': f'data:image/jpeg;base64,{img_base64}',
            'original_image_with_corners': f'data:image/jpeg;base64,{orig_base64}'
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
