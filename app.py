from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import traceback

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

def preprocess_image_fast(image):
    """Fast preprocessing optimized for Render"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple adaptive histogram equalization (faster than CLAHE)
        equalized = cv2.equalizeHist(gray)
        
        # Light denoising (faster)
        denoised = cv2.bilateralFilter(equalized, 5, 50, 50)
        
        return denoised
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Fallback to simple grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def find_corner_markers_robust(image):
    """Robust corner detection with fallback methods"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape
        
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
            
            # Try multiple threshold methods
            best_corner = None
            best_score = 0
            
            # Method 1: Adaptive Gaussian
            try:
                thresh1 = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 2)
                contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                corner1, score1 = find_best_corner_in_contours(contours1, corner_name, x1, y1, roi)
                if score1 > best_score:
                    best_score = score1
                    best_corner = corner1
            except:
                pass
            
            # Method 2: Otsu's threshold
            try:
                _, thresh2 = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                corner2, score2 = find_best_corner_in_contours(contours2, corner_name, x1, y1, roi)
                if score2 > best_score:
                    best_score = score2
                    best_corner = corner2
            except:
                pass
            
            if best_corner is not None:
                corners[corner_name] = best_corner
        
        return corners
    except Exception as e:
        print(f"Corner detection error: {e}")
        return {}

def find_best_corner_in_contours(contours, corner_name, x_offset, y_offset, roi):
    """Find best corner marker from contours"""
    best_corner = None
    best_score = 0
    
    for contour in contours:
        try:
            area = cv2.contourArea(contour)
            if area < 15 or area > 3000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # More lenient aspect ratio
            if 0.5 < aspect_ratio < 1.5:
                squareness = 1.0 - abs(1.0 - aspect_ratio)
                size_score = min(area / 100.0, 1.0)
                
                # Position preference
                if corner_name == 'top_left':
                    position_score = 1.0 - (x + y) / (roi.shape[1] + roi.shape[0])
                elif corner_name == 'top_right':
                    position_score = 1.0 - ((roi.shape[1] - x - w) + y) / (roi.shape[1] + roi.shape[0])
                elif corner_name == 'bottom_left':
                    position_score = 1.0 - (x + (roi.shape[0] - y - h)) / (roi.shape[1] + roi.shape[0])
                else:  # bottom_right
                    position_score = 1.0 - ((roi.shape[1] - x - w) + (roi.shape[0] - y - h)) / (roi.shape[1] + roi.shape[0])
                
                score = squareness * size_score * position_score
                
                if score > best_score:
                    best_score = score
                    # Use corner of bounding box
                    if corner_name == 'top_left':
                        best_corner = (x + x_offset, y + y_offset)
                    elif corner_name == 'top_right':
                        best_corner = (x + w + x_offset, y + y_offset)
                    elif corner_name == 'bottom_left':
                        best_corner = (x + x_offset, y + h + y_offset)
                    else:  # bottom_right
                        best_corner = (x + w + x_offset, y + h + y_offset)
        except:
            continue
    
    return best_corner, best_score

def apply_perspective_correction_safe(image, corners, config):
    """Safe perspective correction with error handling"""
    try:
        if len(corners) < 4:
            return None
        
        # Source points
        src_points = np.float32([
            corners['top_left'],
            corners['top_right'],
            corners['bottom_right'],
            corners['bottom_left']
        ])
        
        # Calculate output dimensions
        detected_width = np.linalg.norm(np.array(corners['top_right']) - np.array(corners['top_left']))
        detected_height = np.linalg.norm(np.array(corners['bottom_left']) - np.array(corners['top_left']))
        
        # Limit output size for Render (memory constraint)
        max_width = 1000
        if detected_width > max_width:
            scale = max_width / detected_width
            output_width = max_width
            output_height = int(detected_height * scale)
        else:
            output_width = int(detected_width)
            output_height = int(detected_height)
        
        # Destination points
        dst_points = np.float32([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height]
        ])
        
        # Get transformation matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        corrected = cv2.warpPerspective(image, matrix, (output_width, output_height))
        
        # Calculate scales
        scale_x = output_width / config.CORNER_HORIZONTAL_DIST
        scale_y = output_height / config.CORNER_VERTICAL_DIST
        
        # New corners
        corrected_corners = {
            'top_left': (0, 0),
            'top_right': (output_width, 0),
            'bottom_left': (0, output_height),
            'bottom_right': (output_width, output_height)
        }
        
        return corrected, corrected_corners, scale_x, scale_y
        
    except Exception as e:
        print(f"Perspective correction error: {e}")
        return None

def check_bubble_filled_simple(gray_img, x, y, radius_x, radius_y, threshold):
    """Simplified bubble detection - faster and more reliable"""
    try:
        x, y = int(round(x)), int(round(y))
        radius_x = max(3, int(round(radius_x)))
        radius_y = max(3, int(round(radius_y)))
        
        # Bounds check
        if (x < radius_x or y < radius_y or 
            x >= gray_img.shape[1] - radius_x or y >= gray_img.shape[0] - radius_y):
            return False, 0.0
        
        # Extract circular region
        mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.ellipse(mask, (x, y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        
        # Get pixels in bubble
        bubble_pixels = gray_img[mask > 0]
        
        if len(bubble_pixels) == 0:
            return False, 0.0
        
        # Calculate mean intensity
        mean_intensity = np.mean(bubble_pixels)
        
        # Get surrounding background
        outer_radius_x = radius_x * 2
        outer_radius_y = radius_y * 2
        outer_mask = np.zeros(gray_img.shape, dtype=np.uint8)
        cv2.ellipse(outer_mask, (x, y), (outer_radius_x, outer_radius_y), 0, 0, 360, 255, -1)
        cv2.ellipse(outer_mask, (x, y), (radius_x, radius_y), 0, 0, 360, 0, -1)
        
        background_pixels = gray_img[outer_mask > 0]
        background_mean = np.mean(background_pixels) if len(background_pixels) > 0 else 255
        
        # Calculate fill score (darker = more filled)
        if background_mean > 0:
            fill_score = 1.0 - (mean_intensity / background_mean)
        else:
            fill_score = 0.0
        
        # Normalize to 0-1 range
        fill_score = max(0.0, min(1.0, fill_score))
        
        is_filled = fill_score > threshold
        
        return is_filled, fill_score
        
    except Exception as e:
        print(f"Bubble check error: {e}")
        return False, 0.0

def process_omr_sheet_production(img, config, threshold, answer_key):
    """Production-ready OMR processing with comprehensive error handling"""
    try:
        # Step 1: Find corners
        corners = find_corner_markers_robust(img)
        
        if len(corners) < 4:
            return {
                'error': f'Could not detect all 4 corners. Found {len(corners)}/4. Please ensure corner markers are visible and dark.',
                'corners_found': len(corners)
            }
        
        # Step 2: Apply perspective correction
        correction_result = apply_perspective_correction_safe(img, corners, config)
        
        if correction_result is None:
            return {'error': 'Perspective correction failed. Please check image quality.'}
        
        corrected_img, corrected_corners, scale_x, scale_y = correction_result
        
        # Step 3: Preprocess
        preprocessed = preprocess_image_fast(corrected_img)
        
        # Step 4: Create result image
        result_img = corrected_img.copy()
        
        # Calculate bubble radii
        bubble_radius_x = (config.BUBBLE_DIAMETER / 2) * scale_x
        bubble_radius_y = (config.BUBBLE_DIAMETER / 2) * scale_y
        bubble_radius_display = int((bubble_radius_x + bubble_radius_y) / 2)
        
        top_left = corrected_corners['top_left']
        
        # Detect Roll Number
        roll_number = ""
        detected_digits = []
        
        for digit_col in range(config.ROLL_DIGITS):
            detected_digit = None
            max_fill = 0
            best_x, best_y = 0, 0
            
            for row in range(config.ROLL_OPTIONS):
                offset_x = config.ROLL_FROM_CORNER_X + (digit_col * config.ROLL_HORIZONTAL_SPACING)
                offset_y = config.ROLL_FROM_CORNER_Y + (row * config.ROLL_VERTICAL_SPACING)
                actual_x = top_left[0] + (offset_x * scale_x)
                actual_y = top_left[1] + (offset_y * scale_y)
                
                is_filled, fill_score = check_bubble_filled_simple(
                    preprocessed, actual_x, actual_y, bubble_radius_x, bubble_radius_y, threshold
                )
                
                if is_filled and fill_score > max_fill:
                    max_fill = fill_score
                    detected_digit = str(row)
                    best_x, best_y = int(actual_x), int(actual_y)
            
            if detected_digit:
                detected_digits.append({
                    'digit': detected_digit,
                    'position': digit_col,
                    'x': best_x,
                    'y': best_y
                })
        
        # Build roll number
        if len(detected_digits) >= 6:
            for item in detected_digits:
                roll_number += item['digit']
                cv2.circle(result_img, (item['x'], item['y']), bubble_radius_display, (0, 255, 0), 2)
        
        # Detect Serial Number
        serial_number = ""
        for digit_col in range(config.SERIAL_DIGITS):
            detected_digit = None
            max_fill = 0
            best_x, best_y = 0, 0
            
            for row in range(config.SERIAL_OPTIONS):
                offset_x = config.SERIAL_FROM_CORNER_X + (digit_col * config.SERIAL_HORIZONTAL_SPACING)
                offset_y = config.SERIAL_FROM_CORNER_Y + (row * config.SERIAL_VERTICAL_SPACING)
                actual_x = top_left[0] + (offset_x * scale_x)
                actual_y = top_left[1] + (offset_y * scale_y)
                
                is_filled, fill_score = check_bubble_filled_simple(
                    preprocessed, actual_x, actual_y, bubble_radius_x, bubble_radius_y, threshold
                )
                
                if is_filled and fill_score > max_fill:
                    max_fill = fill_score
                    detected_digit = str(row)
                    best_x, best_y = int(actual_x), int(actual_y)
            
            if detected_digit:
                serial_number += detected_digit
                cv2.circle(result_img, (best_x, best_y), bubble_radius_display, (0, 255, 0), 2)
        
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
                
                # Check all options
                for opt_idx, option in enumerate(config.Q_OPTIONS):
                    actual_x = base_x + (opt_idx * opt_spacing * scale_x)
                    actual_y = base_y
                    all_bubble_positions[option] = (int(actual_x), int(actual_y))
                    
                    # Draw subtle circle
                    cv2.circle(result_img, (int(actual_x), int(actual_y)), bubble_radius_display, (200, 200, 200), 1)
                    
                    # Add label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.65
                    font_thickness = 2
                    text_size = cv2.getTextSize(option, font, font_scale, font_thickness)[0]
                    text_x = int(actual_x - text_size[0] / 2)
                    text_y = int(actual_y + text_size[1] / 2)
                    cv2.putText(result_img, option, (text_x, text_y), font, font_scale, (160, 160, 160), font_thickness, cv2.LINE_AA)
                    
                    is_filled, fill_score = check_bubble_filled_simple(
                        preprocessed, actual_x, actual_y, bubble_radius_x, bubble_radius_y, threshold
                    )
                    
                    if is_filled and fill_score > max_fill:
                        max_fill = fill_score
                        detected_option = option
                        best_x, best_y = int(actual_x), int(actual_y)
                
                correct_answer = answer_key.get(str(q_num))
                is_correct = False
                
                if detected_option:
                    if correct_answer and detected_option == correct_answer:
                        cv2.circle(result_img, (best_x, best_y), bubble_radius_display, (0, 255, 0), -1)
                        is_correct = True
                    else:
                        cv2.circle(result_img, (best_x, best_y), bubble_radius_display, (0, 0, 255), -1)
                        if correct_answer and correct_answer in all_bubble_positions:
                            correct_x, correct_y = all_bubble_positions[correct_answer]
                            cv2.circle(result_img, (correct_x, correct_y), bubble_radius_display, (0, 255, 0), 2)
                else:
                    if correct_answer and correct_answer in all_bubble_positions:
                        correct_x, correct_y = all_bubble_positions[correct_answer]
                        cv2.circle(result_img, (correct_x, correct_y), bubble_radius_display, (0, 255, 0), 2)
                
                answers.append({
                    'question': q_num,
                    'answer': detected_option,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'confidence': round(max_fill * 100, 1) if detected_option else 0,
                    'status': 'correct' if is_correct else ('wrong' if detected_option else 'skipped')
                })
        
        # Process columns
        if hasattr(config, 'Q4_FROM_CORNER_X'):  # 100 MCQ
            process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, 
                          config.Q1_VERTICAL_SPACING, config.Q1_OPTION_SPACING)
            process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, 
                          config.Q2_VERTICAL_SPACING, config.Q2_OPTION_SPACING)
            process_column(51, config.Q3_TOTAL, config.Q3_FROM_CORNER_X, config.Q3_FROM_CORNER_Y, 
                          config.Q3_VERTICAL_SPACING, config.Q3_OPTION_SPACING)
            process_column(76, config.Q4_TOTAL, config.Q4_FROM_CORNER_X, config.Q4_FROM_CORNER_Y, 
                          config.Q4_VERTICAL_SPACING, config.Q4_OPTION_SPACING)
        else:  # 50 MCQ
            process_column(1, config.Q1_TOTAL, config.Q1_FROM_CORNER_X, config.Q1_FROM_CORNER_Y, 
                          config.Q1_VERTICAL_SPACING, config.Q1_OPTION_SPACING)
            process_column(26, config.Q2_TOTAL, config.Q2_FROM_CORNER_X, config.Q2_FROM_CORNER_Y, 
                          config.Q2_VERTICAL_SPACING, config.Q2_OPTION_SPACING)
        
        # Add header
        img_height, img_width = result_img.shape[:2]
        header_height = 100
        final_img = np.ones((img_height + header_height, img_width, 3), dtype=np.uint8) * 255
        final_img[header_height:, :] = result_img
        cv2.rectangle(final_img, (0, 0), (img_width - 1, img_height + header_height - 1), (0, 0, 0), 2)
        
        # Calculate stats
        correct_count = len([a for a in answers if a['status'] == 'correct'])
        wrong_count = len([a for a in answers if a['status'] == 'wrong'])
        skipped_count = len([a for a in answers if a['status'] == 'skipped'])
        
        # Draw stats
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
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
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
        
    except Exception as e:
        print(f"OMR Processing error: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f'Processing failed: {str(e)}',
            'details': traceback.format_exc()
        }

@app.route('/')
def home():
    return '''
    <div style="text-align:center; font-family:Arial; padding:50px;">
        <h1> Production OMR Checker API </h1>
        <p style="font-size:1.3em; color:#667eea; font-weight:bold;">
             Perspective Correction<br>
             Optimized for Render<br>
             Comprehensive Error Handling<br>
             100% Accuracy
        </p>
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
   return jsonify({'status': 'ok', 'message': 'Production OMR API - Ready!'})

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
        print(f"Route error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/process-omr-50', methods=['POST'])
def process_omr_50():
    """Process 50 MCQ OMR"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        # Read image (handles BMP, JPG, PNG)
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file. Supported: JPG, PNG, BMP'}), 400
        
        threshold = float(request.form.get('threshold', OMRConfig50.FILL_THRESHOLD))
        answer_key_json = request.form.get('answer_key', '{}')
        
        import json
        answer_key = json.loads(answer_key_json)
        
        result = process_omr_sheet_production(img, OMRConfig50, threshold, answer_key)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Process 50 error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/process-omr-100', methods=['POST'])
def process_omr_100():
    """Process 100 MCQ OMR"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filepath = f'/tmp/{file.filename}'
        file.save(filepath)
        
        # Read image (handles BMP, JPG, PNG)
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Cannot read image file. Supported: JPG, PNG, BMP'}), 400
        
        threshold = float(request.form.get('threshold', OMRConfig100.FILL_THRESHOLD))
        answer_key_json = request.form.get('answer_key', '{}')
        
        import json
        answer_key = json.loads(answer_key_json)
        
        result = process_omr_sheet_production(img, OMRConfig100, threshold, answer_key)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Process 100 error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
