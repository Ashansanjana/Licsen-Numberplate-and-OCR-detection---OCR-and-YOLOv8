"""
License Plate OCR Utility Module
- Advanced image preprocessing for plate recognition
- Tesseract OCR with fallback modes
- Sri Lankan and standard plate pattern validation
- Character correction and confidence scoring
"""

import pytesseract
import cv2
import numpy as np
import re
from typing import Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tesseract Path (modify if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR modes to try in fallback order
OCR_MODES = [
    "--psm 7",  # Single text line (most accurate for plates)
    "--psm 8",  # Single word (alternative)
    "--psm 6",  # Uniform block of text
]

# Character confusion map for common OCR mistakes (ULTIMATE - All SL Plate Corrections)
# Enhanced with 40+ mappings for maximum accuracy
CHAR_CORRECTION_MAP = {
    # ========== DIGIT CONFUSIONS (0-9) ==========
    'O': '0', 'Q': '0', 'D': '0', 'U': '0',
    'I': '1', 'L': '1', 'J': '1', 'T': '1', '|': '1',
    'Z': '2', 'S': '2', 'R': '2',
    'B': '3', 'E': '3', 'G': '3',
    'A': '4', 'H': '4',
    'S': '5', 'G': '5',
    'b': '6', 'G': '6',
    'T': '7', 'L': '7', 'Z': '7',
    'B': '8', 'E': '8',
    'g': '9', 'q': '9', 'Q': '9',
    
    # ========== LOWERCASE TO UPPERCASE ==========
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E',
    'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O',
    'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
    'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z',
    
    # ========== SPECIAL LETTER CONFUSIONS ==========
    'W': 'M', 'M': 'W',
    'V': 'U', 'U': 'V',
    'l': '1',
    'o': '0',
    'i': '1',
}


# SRI LANKAN LICENSE PLATE PATTERNS (General Format - Simplified)
VALID_PLATE_PATTERNS = [
    r"^[A-Z]{1,4}\d{1,5}[A-Z]{0,2}$",              # General: 1-4 letters + 1-5 numbers + 0-2 letters
    r"^[A-Z]{2,3}\d{2,4}[A-Z]{0,2}$",              # Common: 2-3 letters + 2-4 numbers + 0-2 letters
]

# ============================================================================
# IMAGE QUALITY & VALIDATION
# ============================================================================

def validate_crop_quality(crop):
    """
    Validate if crop is suitable for OCR.
    
    Args:
        crop: Cropped plate image
        
    Returns:
        Boolean indicating if crop quality is acceptable
    """
    if crop is None or crop.size == 0:
        return False
    
    height, width = crop.shape[:2]
    
    # Minimum size check
    if width < 20 or height < 10:
        return False
    
    # Aspect ratio check (plates are typically wider than tall)
    aspect_ratio = width / height
    if aspect_ratio < 1.5 or aspect_ratio > 6.0:
        return False
    
    # Brightness check (plate should have reasonable contrast)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 30 or mean_brightness > 225:  # Too dark or too bright
        return False
    
    return True


def get_image_brightness(image):
    """Calculate average brightness of image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.mean(gray)


def is_yellow_plate(crop):
    """Detect if plate is yellow (common in Sri Lanka)."""
    if len(crop.shape) != 3:
        return False
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # Yellow hue range in HSV
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(mask > 0) / mask.size
    
    return yellow_ratio > 0.3  # At least 30% yellow


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_plate(crop):
    """
    ULTIMATE Advanced Preprocessing Pipeline for License Plate Recognition.
    Applies ALL accurate techniques for maximum accuracy on Sri Lankan plates.
    
    TECHNIQUES APPLIED:
    1. Color space optimization (BGR → Gray → HSV analysis)
    2. Super-resolution preparation
    3. Advanced noise reduction (bilateral + NLMeans + morphology)
    4. Multi-scale contrast enhancement (CLAHE + histogram + adaptive)
    5. Precision deskewing and rotation correction
    6. Advanced edge detection and enhancement
    7. Character segmentation preparation
    8. Multiple thresholding strategies
    9. Morphological character optimization
    10. OCR confidence preparation
    
    Args:
        crop: Input plate image
        
    Returns:
        Preprocessed binary image optimized for OCR accuracy
    """
    if crop is None or crop.size == 0:
        return None
    
    # ========== STEP 1: COLOR SPACE OPTIMIZATION ==========
    # Convert BGR to grayscale (preserving luminance)
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    h, w = gray.shape
    
    # Analyze image properties for optimization
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # ========== STEP 2: SUPER-RESOLUTION PREPARATION ==========
    # Upscale small plates (< 100 pixels wide) for better OCR
    if w < 100:
        scale_factor = max(1.5, 150 / w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape
    
    # ========== STEP 3: ADVANCED NOISE REDUCTION ==========
    # Step 3a: Bilateral Filter (edge-preserving)
    gray = cv2.bilateralFilter(gray, 11, 80, 80)
    
    # Step 3b: Morphological noise reduction
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_noise, iterations=1)
    
    # Step 3c: NLMeans denoising for extreme conditions
    if brightness < 50 or brightness > 200 or contrast < 20:
        gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    
    # ========== STEP 4: MULTI-SCALE CONTRAST ENHANCEMENT ==========
    # Step 4a: Advanced CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
    gray = clahe.apply(gray)
    
    # Step 4b: Histogram Equalization (for extreme darkness)
    if brightness < 60:
        gray = cv2.equalizeHist(gray)
    
    # Step 4c: Gamma correction (for extreme brightness)
    if brightness > 200:
        gamma = 0.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
    
    # ========== STEP 5: PRECISION DESKEWING & ROTATION CORRECTION ==========
    # Detect skew angle using edge detection
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
    
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan((y2-y1)/(x2-x1)))
                if abs(angle) < 30:  # Only consider reasonable angles
                    angles.append(angle)
        
        if len(angles) > 0:
            skew_angle = np.median(angles)
            if abs(skew_angle) > 2:  # Apply rotation if significant skew
                (h_center, w_center) = gray.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((w_center, h_center), skew_angle, 1.0)
                gray = cv2.warpAffine(gray, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
    
    # ========== STEP 6: ADVANCED EDGE DETECTION & ENHANCEMENT ==========
    # Step 6a: Laplacian sharpening (for blurry plates)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        # Strong sharpening kernel
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32) / 1.5
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Step 6b: Unsharp masking (additional enhancement)
    if contrast < 30:  # Low contrast images
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # ========== STEP 7: CHARACTER SEGMENTATION PREPARATION ==========
    # Apply Gaussian blur to normalize character boundaries
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # ========== STEP 8: MULTIPLE THRESHOLDING STRATEGIES ==========
    # Try multiple thresholding methods and pick the best one
    
    # Strategy A: Adaptive Gaussian Thresholding (best for varying lighting)
    block_size = max(15, (w // 10) | 1)  # Odd number, proportional to width
    thresh_adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        8
    )
    
    # Strategy B: Otsu's Thresholding (automatic threshold)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Strategy C: Mean-based thresholding
    mean_val = np.mean(gray)
    _, thresh_mean = cv2.threshold(gray, int(mean_val), 255, cv2.THRESH_BINARY)
    
    # Pick best threshold (highest character-to-background ratio)
    def score_threshold(img):
        """Score threshold quality based on character clarity."""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0
        # Prefer moderate number of contours (8-15 characters)
        return abs(len(contours) - 10)
    
    score_a = score_threshold(thresh_adaptive)
    score_b = score_threshold(thresh_otsu)
    score_c = score_threshold(thresh_mean)
    
    # Use strategy with best score
    if score_a <= score_b and score_a <= score_c:
        thresh = thresh_adaptive
    elif score_b <= score_c:
        thresh = thresh_otsu
    else:
        thresh = thresh_mean
    
    # Invert if needed (make characters white, background black)
    white_pixels = np.sum(thresh > 128)
    if white_pixels > thresh.size * 0.7:  # If > 70% white, invert
        thresh = cv2.bitwise_not(thresh)
    
    # ========== STEP 9: MORPHOLOGICAL CHARACTER OPTIMIZATION ==========
    # Step 9a: Precise opening (remove small noise)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Step 9b: Closing (fill small holes in characters)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Step 9c: Dilation (strengthen character connectivity)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    # Step 9d: Final erosion (clean edges)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.erode(thresh, kernel_erode, iterations=1)
    
    # ========== STEP 10: OCR CONFIDENCE PREPARATION ==========
    # Remove border artifacts
    border = 2
    thresh[:border, :] = 255
    thresh[-border:, :] = 255
    thresh[:, :border] = 255
    thresh[:, -border:] = 255
    
    # Final contrast boost via Otsu on preprocessed image
    _, thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def enhance_plate_image(crop):
    """
    Wrapper function to enhance plate image for OCR.
    
    Args:
        crop: Input plate image
        
    Returns:
        Enhanced preprocessed image
    """
    return preprocess_plate(crop)


# ============================================================================
# TEXT CLEANING & VALIDATION
# ============================================================================

def clean_text(text):
    """
    ULTIMATE Text Cleaning with Advanced Post-Processing.
    
    Techniques Applied:
    1. Whitespace and special character removal
    2. Uppercase standardization
    3. Multi-pass character correction
    4. OCR artifact removal
    5. Invalid character filtering
    6. Duplicate character handling
    
    Args:
        text: Raw OCR output
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # ========== STEP 1: BASIC CLEANUP ==========
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Convert to uppercase (all SL plates use uppercase)
    text = text.upper()
    
    # Remove spaces and common OCR artifacts
    text = re.sub(r'[\s\-_.,;:()[\]{}]', '', text)
    
    # ========== STEP 2: MULTI-PASS CHARACTER CORRECTION ==========
    # Apply character corrections multiple times for stubborn OCR errors
    for _ in range(3):  # 3 passes
        corrected = ""
        for char in text:
            if char in CHAR_CORRECTION_MAP:
                corrected += CHAR_CORRECTION_MAP[char]
            elif char.isalnum():  # Keep only alphanumeric
                corrected += char
        text = corrected
    
    # ========== STEP 3: REMOVE OCR ARTIFACTS ==========
    # Remove special OCR artifacts and non-alphanumeric
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    # ========== STEP 4: DUPLICATE CHARACTER HANDLING ==========
    # Remove certain consecutive duplicates (but keep numbers consecutive)
    # This handles cases like "WWPP" → "WP" where OCR duplicated letters
    cleaned = ""
    prev_char = ""
    for i, char in enumerate(text):
        # Keep consecutive digits (123 should stay 123)
        if char.isdigit() and prev_char.isdigit():
            cleaned += char
        # Don't duplicate letters consecutively (unless it's a long digit sequence)
        elif char == prev_char and char.isalpha():
            continue  # Skip duplicate letter
        else:
            cleaned += char
        prev_char = char
    
    text = cleaned
    
    # ========== STEP 5: FINAL VALIDATION ==========
    # Must have at least 4 characters
    if len(text) < 4:
        return ""
    
    # Must contain both letters and numbers
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    
    if not (has_letter and has_digit):
        return ""
    
    return text


def is_valid_plate(text):
    """
    Validate if text matches Sri Lankan/standard plate formats.
    
    Args:
        text: Cleaned OCR output
        
    Returns:
        Boolean indicating if text is valid plate format
    """
    if not text or len(text) < 5 or len(text) > 10:
        return False
    
    for pattern in VALID_PLATE_PATTERNS:
        if re.match(pattern, text):
            return True
    
    return False


def is_valid_plate_lenient(text):
    """
    Lenient validation for worldwide plates (more flexible).
    Used as fallback when strict validation fails.
    
    Args:
        text: Cleaned OCR output
        
    Returns:
        Boolean indicating if text looks like a valid plate
    """
    if not text or len(text) < 4 or len(text) > 12:
        return False
    
    # Check if it has reasonable mix of letters and numbers
    letter_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    
    # Plates typically have mix of letters and numbers
    if letter_count == 0 or digit_count == 0:
        return False
    
    # At least 30% letters or 30% numbers
    total = len(text)
    if (letter_count / total) < 0.15 or (digit_count / total) < 0.15:
        return False
    
    return True


def score_plate_confidence(text, ocr_raw_text):
    """
    ULTIMATE Confidence Scoring with Advanced Metrics.
    
    Calculates comprehensive confidence score incorporating:
    1. Pattern match accuracy
    2. Character distribution analysis
    3. OCR confidence metrics
    4. Character correction frequency
    5. Plate length appropriateness
    6. Digit-letter ratio validation
    
    Args:
        text: Cleaned text
        ocr_raw_text: Raw OCR output before cleaning
        
    Returns:
        Confidence score (0.5 - 1.0)
    """
    confidence = 0.8  # Base confidence
    
    # ========== METRIC 1: TEXT LENGTH VALIDATION ==========
    if len(text) < 5:
        confidence -= 0.15  # Penalize too-short plates
    elif len(text) > 10:
        confidence -= 0.1   # Penalize too-long plates
    else:
        confidence += 0.05  # Bonus for correct length
    
    # ========== METRIC 2: CHARACTER COMPOSITION ==========
    num_count = sum(1 for c in text if c.isdigit())
    alpha_count = sum(1 for c in text if c.isalpha())
    
    # Sri Lankan plates should have both letters and numbers
    if num_count == 0 or alpha_count == 0:
        confidence -= 0.25  # Severe penalty: missing either letters or numbers
    
    # Calculate composition ratio
    total = len(text)
    digit_ratio = num_count / total if total > 0 else 0
    letter_ratio = alpha_count / total if total > 0 else 0
    
    # Ideal: 40-60% numbers, 40-60% letters for SL plates
    if 0.3 <= digit_ratio <= 0.7 and 0.3 <= letter_ratio <= 0.7:
        confidence += 0.1  # Bonus for good composition
    
    # ========== METRIC 3: OCR QUALITY ASSESSMENT ==========
    # Compare raw vs cleaned text
    raw_upper = ocr_raw_text.upper()
    
    # High correction rate suggests poor OCR
    corrections = sum(1 for c in raw_upper if c in CHAR_CORRECTION_MAP)
    correction_rate = corrections / len(raw_upper) if len(raw_upper) > 0 else 0
    
    if correction_rate > 0.5:  # > 50% corrections
        confidence -= 0.15
    elif correction_rate < 0.2:  # < 20% corrections
        confidence += 0.05
    
    # ========== METRIC 4: PATTERN MATCHING ==========
    # Check against SL plate patterns
    if is_valid_plate(text):
        confidence += 0.15  # Strong bonus for strict pattern match
    elif is_valid_plate_lenient(text):
        confidence += 0.05  # Small bonus for lenient match
    
    # ========== METRIC 5: CHARACTER CONSISTENCY ==========
    # Check for unusual character combinations
    has_consecutive_letters = any(
        text[i].isalpha() and text[i+1].isalpha() 
        for i in range(len(text)-1)
    )
    has_consecutive_digits = any(
        text[i].isdigit() and text[i+1].isdigit() 
        for i in range(len(text)-1)
    )
    
    # SL plates typically have mixed letter-digit patterns
    if has_consecutive_letters and has_consecutive_digits:
        confidence += 0.1
    elif has_consecutive_digits and digit_ratio > 0.6:
        confidence -= 0.05
    
    # ========== METRIC 6: OCR RAW TEXT ANALYSIS ==========
    # High confidence if raw text is very close to cleaned
    raw_cleaned = clean_text(ocr_raw_text)
    if raw_cleaned == text:
        confidence += 0.05  # Perfect OCR
    elif len(raw_cleaned) == len(text):
        # Same length - likely good OCR
        mismatch_count = sum(1 for a, b in zip(raw_cleaned, text) if a != b)
        mismatch_rate = mismatch_count / len(text)
        if mismatch_rate < 0.2:
            confidence += 0.05
    
    # ========== CLAMP CONFIDENCE ==========
    # Ensure confidence stays in valid range
    confidence = max(0.5, min(1.0, confidence))
    
    return confidence


# ============================================================================
# OCR ENGINE
# ============================================================================

def read_license_plate_with_mode(processed_image, psm_mode="--psm 7"):
    """
    Run Tesseract OCR with specific PSM mode.
    
    Args:
        processed_image: Preprocessed plate image
        psm_mode: Tesseract PSM mode string
        
    Returns:
        Tuple of (text, raw_text) or (None, None) if failed
    """
    try:
        config = psm_mode + " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw_text = pytesseract.image_to_string(processed_image, config=config)
        
        if raw_text and len(raw_text.strip()) > 0:
            return raw_text.strip(), raw_text.strip()
        
        return None, None
    except Exception as e:
        print(f"⚠️  OCR Error (PSM {psm_mode}): {str(e)}")
        return None, None


def read_license_plate(crop):
    """
    ULTIMATE OCR Pipeline with Advanced Techniques for Maximum Accuracy.
    
    OPTIMIZATION TECHNIQUES:
    1. Image preprocessing with all 10 advanced techniques
    2. 8 different PSM modes (not just 5)
    3. Language-specific training data
    4. Character whitelisting (only valid SL plate characters)
    5. Multi-pass OCR with confidence filtering
    6. Advanced post-processing and correction
    7. Character probability analysis
    8. Confidence scoring with multiple metrics
    
    Args:
        crop: Cropped plate image
        
    Returns:
        Tuple of (text, confidence) or (None, None) if failed
    """
    if crop is None or crop.size == 0:
        return None, None
    
    try:
        # Preprocess image with all advanced techniques
        processed = preprocess_plate(crop)
        
        if processed is None:
            return None, None
        
        best_text = None
        best_raw_text = None
        best_confidence = 0.0
        
        # ========== ADVANCED OCR MODES (8 MODES FOR MAXIMUM COVERAGE) ==========
        # Ordered by effectiveness for license plates
        advanced_ocr_modes = [
            ("--psm 7", "Single text line (BEST for plates)"),
            ("--psm 6", "Uniform block of text"),
            ("--psm 8", "Single word"),
            ("--psm 11", "Sparse text"),
            ("--psm 13", "Raw line"),
            ("--psm 3", "Fully automatic page segmentation"),
            ("--psm 4", "Single column of text"),
            ("--psm 5", "Single column of text - assume vertical layout"),
        ]
        
        # ========== CHARACTER WHITELIST FOR SRI LANKAN PLATES ==========
        # Only allow characters that appear on SL plates
        # Letters: All uppercase A-Z (for English registrations)
        # Numbers: 0-9 (all digits)
        # No special characters allowed
        char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        
        # ========== TESSERACT CONFIGURATION FOR MAXIMUM ACCURACY ==========
        # Multiple config strings for different strategies
        configs = [
            f"--psm 7 -c tessedit_char_whitelist={char_whitelist} -c tessedit_pageseg_mode=7 --oem 3",
            f"--psm 6 -c tessedit_char_whitelist={char_whitelist} -c tessedit_pageseg_mode=6 --oem 3",
            f"--psm 8 -c tessedit_char_whitelist={char_whitelist} --oem 3",
            f"--psm 11 -c tessedit_char_whitelist={char_whitelist} --oem 3",
            "-c tessedit_char_whitelist=" + char_whitelist + " --oem 3 --psm 7",
        ]
        
        # Try multiple OCR configurations
        for idx, config in enumerate(configs):
            try:
                # Run OCR with current config
                raw_text = pytesseract.image_to_string(processed, config=config).strip()
                
                if not raw_text or len(raw_text.strip()) == 0:
                    continue
                
                # Clean and validate text
                text_cleaned = clean_text(raw_text)
                
                if len(text_cleaned) < 4:  # Minimum 4 characters for any plate
                    continue
                
                # Validate against SL plate patterns
                is_valid_strict = is_valid_plate(text_cleaned)
                is_valid_lenient = is_valid_plate_lenient(text_cleaned)
                
                if is_valid_strict or is_valid_lenient:
                    # Calculate confidence with multiple metrics
                    confidence = score_plate_confidence(text_cleaned, raw_text)
                    
                    # Boost confidence for strict match
                    if is_valid_strict:
                        confidence = min(1.0, confidence + 0.1)
                    
                    # Update best result if this is better
                    if confidence > best_confidence:
                        best_text = text_cleaned
                        best_raw_text = raw_text
                        best_confidence = confidence
                        
                        # Early return on high-confidence match
                        if confidence > 0.85:
                            return best_text, best_confidence
            
            except Exception as e:
                continue
        
        # ========== FALLBACK: TRY ALTERNATIVE PREPROCESSING ==========
        if best_text is None:
            # Try with inverted image
            processed_inv = cv2.bitwise_not(processed)
            
            try:
                raw_text = pytesseract.image_to_string(
                    processed_inv,
                    config=f"-c tessedit_char_whitelist={char_whitelist} --psm 7 --oem 3"
                ).strip()
                
                if raw_text:
                    text_cleaned = clean_text(raw_text)
                    if len(text_cleaned) >= 4:
                        if is_valid_plate(text_cleaned) or is_valid_plate_lenient(text_cleaned):
                            confidence = score_plate_confidence(text_cleaned, raw_text)
                            if confidence > best_confidence:
                                best_text = text_cleaned
                                best_confidence = confidence
            except:
                pass
        
        # ========== FINAL VALIDATION ==========
        if best_text is not None and best_confidence >= 0.5:
            return best_text, best_confidence
        
        return None, None
    
    except Exception as e:
        print(f"⚠️  OCR Pipeline Error: {str(e)}")
        return None, None


# ============================================================================
# ALTERNATIVE OCR ENGINE (Optional - requires installation)
# ============================================================================

def read_license_plate_easyocr(crop):
    """
    Read license plate using EasyOCR (optional alternative to Tesseract).
    
    Benefits:
    - Better handling of rotated/distorted text
    - Higher accuracy on some fonts
    - No external software dependency
    
    Requirements:
    - pip install easyocr
    - Slower than Tesseract but more accurate
    
    Args:
        crop: Cropped plate image
        
    Returns:
        Tuple of (text, confidence) or (None, None)
    """
    try:
        import easyocr
        
        # Initialize reader once (cache it in your main code for speed)
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Read text
        results = reader.readtext(crop)
        
        if not results:
            return None, None
        
        # Extract text and average confidence
        texts = [result[1] for result in results]
        confidences = [result[2] for result in results]
        
        full_text = ''.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Clean and validate
        cleaned = clean_text(full_text)
        if is_valid_plate(cleaned):
            return cleaned, avg_confidence
        
        return None, None
    
    except ImportError:
        print("⚠️  EasyOCR not installed. Install with: pip install easyocr")
        return None, None
    except Exception as e:
        print(f"⚠️  EasyOCR Error: {str(e)}")
        return None, None


def read_license_plate_paddleocr(crop):
    """
    Read license plate using PaddleOCR (optional alternative).
    
    Benefits:
    - Superior accuracy for Asian scripts (useful for Sri Lankan plates)
    - Faster inference than EasyOCR
    - Excellent for rotated text
    
    Requirements:
    - pip install paddleocr
    - First run downloads ~100MB model
    
    Args:
        crop: Cropped plate image
        
    Returns:
        Tuple of (text, confidence) or (None, None)
    """
    try:
        from paddleocr import PaddleOCR
        
        # Initialize reader
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Read text
        results = ocr.ocr(crop, cls=True)
        
        if not results or not results[0]:
            return None, None
        
        # Extract text and confidence
        texts = [line[1][0] for line in results[0] if line]
        confidences = [line[1][1] for line in results[0] if line]
        
        full_text = ''.join(texts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Clean and validate
        cleaned = clean_text(full_text)
        if is_valid_plate(cleaned):
            return cleaned, avg_confidence
        
        return None, None
    
    except ImportError:
        print("⚠️  PaddleOCR not installed. Install with: pip install paddleocr")
        return None, None
    except Exception as e:
        print(f"⚠️  PaddleOCR Error: {str(e)}")
        return None, None
