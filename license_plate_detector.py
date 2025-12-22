"""
=============================================================================
ADVANCED LICENSE PLATE DETECTION & RECOGNITION SYSTEM
Sri Lankan License Plate Detector with YOLO + EasyOCR
=============================================================================

Features:
- YOLO-based license plate detection
- EasyOCR for text recognition
- Real-time video processing
- Output video generation with annotations
- Confidence scoring and visualization
- Sri Lankan plate format validation
- FPS counter and performance metrics

Usage:
    python license_plate_detector.py --input video.mp4 --output results.mp4

Author: Advanced AI
Date: 2024
=============================================================================
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import easyocr
from ultralytics import YOLO
import re
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION - MODIFY HERE FOR YOUR SETUP
# =============================================================================

class Config:
    """Configuration class for all system parameters."""
    
    # ========== FILE PATHS ==========
    # YOLO model path (change to your model)
    YOLO_MODEL_PATH = "best.pt"  # Use custom trained model
    YOLO_FALLBACK_MODEL = "yolov8n.pt"  # Fallback to nano model
    
    # Video input/output paths
    DEFAULT_INPUT_VIDEO = "test_video.mp4"  # Change to your video
    DEFAULT_OUTPUT_VIDEO = "output_detected.mp4"
    
    # ========== DETECTION PARAMETERS ==========
    # Confidence threshold for YOLO detections
    YOLO_CONFIDENCE = 0.4  # Lower for more detections, higher for accuracy
    
    # IOU (Intersection over Union) for NMS
    YOLO_IOU = 0.5
    
    # Minimum plate size (width, height) in pixels
    MIN_PLATE_WIDTH = 20
    MIN_PLATE_HEIGHT = 10
    
    # Maximum plate size (width, height) in pixels
    MAX_PLATE_WIDTH = 500
    MAX_PLATE_HEIGHT = 150
    
    # ========== PREPROCESSING PARAMETERS ==========
    # Padding around detected bounding box
    CROP_PADDING = 5
    
    # Resize target width for OCR (larger = better for small plates)
    RESIZE_TARGET_WIDTH = 200
    
    # ========== OCR PARAMETERS ==========
    # EasyOCR GPU usage (True if CUDA available, False for CPU)
    USE_GPU = True
    
    # EasyOCR language (English for SL plates)
    OCR_LANGUAGE = ['en']
    
    # OCR confidence threshold
    OCR_CONFIDENCE_THRESHOLD = 0.3
    
    # ========== DISPLAY PARAMETERS ==========
    # Frame display size
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720
    
    # Font parameters for text display
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    
    # Color codes (BGR format)
    COLOR_GREEN = (0, 255, 0)      # Detection box
    COLOR_RED = (0, 0, 255)        # Error/low confidence
    COLOR_BLUE = (255, 0, 0)       # Text background
    COLOR_YELLOW = (0, 255, 255)   # Warnings
    COLOR_WHITE = (255, 255, 255)  # Text
    
    # ========== VIDEO PARAMETERS ==========
    # Video codec (mp4v for MP4, XVID for AVI)
    VIDEO_CODEC = 'mp4v'
    
    # Video FPS (frames per second)
    VIDEO_FPS = 30
    
    # ========== SRI LANKAN PLATE VALIDATION ==========
    # Regex patterns for valid SL plates (simplified to common formats)
    SL_PLATE_PATTERNS = [
        r"^[A-Z]{1,4}\d{1,5}[A-Z]{0,2}$",              # General format
        r"^[A-Z]{2,3}\d{3,4}[A-Z]{0,2}$",              # Common format
    ]
    
    # ========== SYSTEM PARAMETERS ==========
    # Maximum frames to process (None = all frames)
    MAX_FRAMES = None  # Set to 100 for testing
    
    # Frame skip rate (1 = every frame, 2 = every 2nd frame)
    FRAME_SKIP = 1
    
    # Enable real-time display window
    SHOW_DISPLAY = True
    
    # Save detected plates to separate folder
    SAVE_PLATES = False
    PLATES_OUTPUT_DIR = "detected_plates"
    
    # Enable verbose logging
    VERBOSE = True


# =============================================================================
# LOGGER CLASS - FOR CONSOLE OUTPUT
# =============================================================================

class Logger:
    """Simple logging utility."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def info(self, msg):
        """Log info message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ℹ️  {msg}")
    
    def success(self, msg):
        """Log success message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ {msg}")
    
    def warning(self, msg):
        """Log warning message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ⚠️  {msg}")
    
    def error(self, msg):
        """Log error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ❌ {msg}")


# =============================================================================
# OCREADER CLASS - MANAGES EASYOCR
# =============================================================================

class OCRReader:
    """Wrapper for EasyOCR with optimization."""
    
    def __init__(self, languages=['en'], use_gpu=True, logger=None):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of languages to recognize
            use_gpu: Whether to use GPU acceleration
            logger: Logger instance
        """
        self.logger = logger or Logger()
        self.use_gpu = use_gpu
        
        try:
            self.logger.info(f"Initializing EasyOCR (GPU={use_gpu})...")
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
            self.logger.success("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            raise
    
    def read_text(self, image, confidence_threshold=0.3):
        """
        Read text from image using EasyOCR.
        
        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence score
            
        Returns:
            Tuple of (detected_text, confidence, raw_results)
        """
        try:
            # Run OCR
            results = self.reader.readtext(image)
            
            if not results:
                return "", 0.0, []
            
            # Extract text and confidence
            detected_text = ""
            confidences = []
            filtered_results = []
            
            for detection in results:
                # Each detection: (bbox, text, confidence)
                text = detection[1]
                confidence = detection[2]
                
                if confidence >= confidence_threshold:
                    detected_text += text
                    confidences.append(confidence)
                    filtered_results.append(detection)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return detected_text.strip(), avg_confidence, filtered_results
        
        except Exception as e:
            return "", 0.0, []


# =============================================================================
# PLATE DETECTOR CLASS - MANAGES YOLO AND PROCESSING
# =============================================================================

class PlateDetector:
    """License plate detection and recognition system."""
    
    def __init__(self, config=None, logger=None):
        """
        Initialize plate detector.
        
        Args:
            config: Configuration class with all parameters
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or Logger(verbose=self.config.VERBOSE)
        
        # Initialize YOLO model
        self._init_yolo_model()
        
        # Initialize OCR reader
        self._init_ocr_reader()
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'plates_detected': 0,
            'successful_ocr': 0,
            'failed_ocr': 0,
            'total_time': 0,
        }
        
        # Create output directory if needed
        if self.config.SAVE_PLATES:
            os.makedirs(self.config.PLATES_OUTPUT_DIR, exist_ok=True)
    
    def _init_yolo_model(self):
        """Initialize YOLO model for plate detection."""
        try:
            # Try to load custom model first
            if os.path.exists(self.config.YOLO_MODEL_PATH):
                self.logger.info(f"Loading YOLO model: {self.config.YOLO_MODEL_PATH}")
                self.yolo_model = YOLO(self.config.YOLO_MODEL_PATH)
                self.logger.success(f"YOLO model loaded: {self.config.YOLO_MODEL_PATH}")
            else:
                # Fall back to pretrained model
                self.logger.warning(f"Model {self.config.YOLO_MODEL_PATH} not found")
                self.logger.info(f"Loading fallback model: {self.config.YOLO_FALLBACK_MODEL}")
                self.yolo_model = YOLO(self.config.YOLO_FALLBACK_MODEL)
                self.logger.success(f"Fallback YOLO model loaded: {self.config.YOLO_FALLBACK_MODEL}")
        
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def _init_ocr_reader(self):
        """Initialize EasyOCR reader."""
        try:
            self.ocr_reader = OCRReader(
                languages=self.config.OCR_LANGUAGE,
                use_gpu=self.config.USE_GPU,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {str(e)}")
            raise
    
    def detect_plates(self, frame):
        """
        Detect license plates in frame using YOLO.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detections [(x1, y1, x2, y2, confidence, class_id)]
        """
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=self.config.YOLO_CONFIDENCE, iou=self.config.YOLO_IOU)
            
            detections = []
            
            # Extract detections
            for detection in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                confidence = detection.conf[0].cpu().item()
                class_id = int(detection.cls[0].cpu().item())
                
                # Validate plate size
                width = x2 - x1
                height = y2 - y1
                
                if (width >= self.config.MIN_PLATE_WIDTH and 
                    height >= self.config.MIN_PLATE_HEIGHT and
                    width <= self.config.MAX_PLATE_WIDTH and
                    height <= self.config.MAX_PLATE_HEIGHT):
                    
                    detections.append((x1, y1, x2, y2, confidence, class_id))
            
            return detections
        
        except Exception as e:
            self.logger.warning(f"Error during plate detection: {str(e)}")
            return []
    
    def crop_plate(self, frame, x1, y1, x2, y2):
        """
        Crop license plate region from frame with padding.
        
        Args:
            frame: Input frame
            x1, y1, x2, y2: Bounding box coordinates
            
        Returns:
            Cropped plate image
        """
        h, w = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - self.config.CROP_PADDING)
        y1 = max(0, y1 - self.config.CROP_PADDING)
        x2 = min(w, x2 + self.config.CROP_PADDING)
        y2 = min(h, y2 + self.config.CROP_PADDING)
        
        crop = frame[y1:y2, x1:x2]
        return crop
    
    def preprocess_plate(self, crop):
        """
        Preprocess cropped plate for OCR.
        
        Args:
            crop: Cropped plate image
            
        Returns:
            Preprocessed image
        """
        if crop is None or crop.size == 0:
            return crop
        
        # Resize if too small
        h, w = crop.shape[:2]
        if w < self.config.RESIZE_TARGET_WIDTH:
            scale = self.config.RESIZE_TARGET_WIDTH / w
            new_w = self.config.RESIZE_TARGET_WIDTH
            new_h = int(h * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply preprocessing
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive thresholding
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21, 10
        )
        
        return enhanced
    
    def recognize_plate(self, crop):
        """
        Recognize license plate text using OCR.
        
        Args:
            crop: Cropped plate image
            
        Returns:
            Tuple of (text, confidence)
        """
        if crop is None or crop.size == 0:
            return "", 0.0
        
        # Preprocess plate
        preprocessed = self.preprocess_plate(crop)
        
        # Read text using EasyOCR
        text, confidence, _ = self.ocr_reader.read_text(
            preprocessed,
            confidence_threshold=self.config.OCR_CONFIDENCE_THRESHOLD
        )
        
        # Clean and validate text
        text_clean = self._clean_text(text)
        
        return text_clean, confidence
    
    def _clean_text(self, text):
        """
        Clean OCR output.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove spaces and special characters
        text = text.upper().strip()
        text = re.sub(r'[\s\-_.,;:()[\]{}]', '', text)
        
        # Keep only alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Must have at least 4 characters
        if len(text) < 4:
            return ""
        
        return text
    
    def is_valid_sri_lankan_plate(self, text):
        """
        Validate if text matches Sri Lankan plate format.
        
        Args:
            text: Cleaned plate text
            
        Returns:
            Boolean indicating validity
        """
        if not text:
            return False
        
        for pattern in self.config.SL_PLATE_PATTERNS:
            if re.match(pattern, text):
                return True
        
        return False
    
    def process_frame(self, frame, frame_id=0):
        """
        Process single video frame.
        
        Args:
            frame: Video frame
            frame_id: Frame number
            
        Returns:
            Tuple of (processed_frame, detections_list)
        """
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        detections = []
        
        # Detect license plates
        plates = self.detect_plates(frame)
        
        if len(plates) > 0:
            for x1, y1, x2, y2, conf, cls_id in plates:
                # Crop and recognize plate
                crop = self.crop_plate(frame, x1, y1, x2, y2)
                text, ocr_conf = self.recognize_plate(crop)
                
                # Validate plate format
                is_valid = self.is_valid_sri_lankan_plate(text)
                
                # Store detection info
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'text': text,
                    'detection_conf': conf,
                    'ocr_conf': ocr_conf,
                    'is_valid': is_valid,
                    'frame_id': frame_id
                })
                
                # Update statistics
                self.stats['plates_detected'] += 1
                if text:
                    self.stats['successful_ocr'] += 1
                else:
                    self.stats['failed_ocr'] += 1
                
                # Draw bounding box
                color = self.config.COLOR_GREEN if is_valid else self.config.COLOR_YELLOW
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                label = f"{text}" if text else "?"
                label += f" ({conf:.2f})"
                
                # Draw text background
                text_size = cv2.getTextSize(label, self.config.FONT, 
                                           self.config.FONT_SCALE, self.config.FONT_THICKNESS)[0]
                text_x = x1
                text_y = max(y1 - 10, 30)
                
                cv2.rectangle(frame_copy, 
                            (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5),
                            self.config.COLOR_BLUE, -1)
                
                cv2.putText(frame_copy, label, (text_x, text_y),
                           self.config.FONT, self.config.FONT_SCALE,
                           self.config.COLOR_WHITE, self.config.FONT_THICKNESS)
                
                # Save individual plate if enabled
                if self.config.SAVE_PLATES and text:
                    self._save_plate_image(crop, text, frame_id)
        
        # Add frame info
        info_text = f"Frame: {frame_id} | Plates: {len(detections)}"
        cv2.putText(frame_copy, info_text, (10, 30),
                   self.config.FONT, self.config.FONT_SCALE,
                   self.config.COLOR_GREEN, self.config.FONT_THICKNESS)
        
        self.stats['total_frames'] += 1
        
        return frame_copy, detections
    
    def _save_plate_image(self, crop, text, frame_id):
        """Save detected plate image to file."""
        try:
            filename = f"{self.config.PLATES_OUTPUT_DIR}/frame{frame_id}_{text}.jpg"
            cv2.imwrite(filename, crop)
        except Exception as e:
            self.logger.warning(f"Failed to save plate image: {str(e)}")
    
    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total Frames Processed:  {self.stats['total_frames']}")
        print(f"Plates Detected:         {self.stats['plates_detected']}")
        print(f"Successful OCR:          {self.stats['successful_ocr']}")
        print(f"Failed OCR:              {self.stats['failed_ocr']}")
        
        if self.stats['plates_detected'] > 0:
            success_rate = (self.stats['successful_ocr'] / self.stats['plates_detected']) * 100
            print(f"OCR Success Rate:        {success_rate:.1f}%")
        
        print("="*60 + "\n")


# =============================================================================
# VIDEO PROCESSOR CLASS - HANDLES VIDEO I/O
# =============================================================================

class VideoProcessor:
    """Handles video file reading and writing."""
    
    def __init__(self, input_path, output_path, config=None, logger=None):
        """
        Initialize video processor.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            config: Configuration class
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or Logger(verbose=self.config.VERBOSE)
        self.input_path = input_path
        self.output_path = output_path
        
        # Open input video
        self.cap = None
        self.writer = None
        self.fps = 30
        self.frame_count = 0
        
        self._init_input_video()
    
    def _init_input_video(self):
        """Initialize input video capture."""
        try:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input video not found: {self.input_path}")
            
            self.cap = cv2.VideoCapture(self.input_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.input_path}")
            
            # Get video properties
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.success(f"Input video opened: {self.input_path}")
            self.logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.frame_count}")
        
        except Exception as e:
            self.logger.error(f"Failed to open input video: {str(e)}")
            raise
    
    def init_output_video(self):
        """Initialize output video writer."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.config.VIDEO_FPS,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                raise ValueError("Failed to initialize VideoWriter")
            
            self.logger.success(f"Output video initialized: {self.output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize output video: {str(e)}")
            raise
    
    def process(self, detector):
        """
        Process entire video with plate detector.
        
        Args:
            detector: PlateDetector instance
        """
        try:
            self.init_output_video()
            
            frame_id = 0
            processed_count = 0
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Check frame limit
                if self.config.MAX_FRAMES and processed_count >= self.config.MAX_FRAMES:
                    break
                
                # Skip frames
                if frame_id % self.config.FRAME_SKIP != 0:
                    frame_id += 1
                    continue
                
                # Process frame
                processed_frame, detections = detector.process_frame(frame, frame_id)
                
                # Write to output video
                self.writer.write(processed_frame)
                
                # Display frame
                if self.config.SHOW_DISPLAY:
                    display_frame = cv2.resize(processed_frame, 
                                              (self.config.DISPLAY_WIDTH, self.config.DISPLAY_HEIGHT))
                    cv2.imshow('License Plate Detection', display_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.warning("Stopped by user")
                        break
                
                # Progress info
                progress = (frame_id / self.frame_count) * 100 if self.frame_count > 0 else 0
                if processed_count % 30 == 0:  # Log every 30 frames
                    self.logger.info(f"Progress: {progress:.1f}% | Frame: {frame_id}/{self.frame_count}")
                
                frame_id += 1
                processed_count += 1
            
            # Cleanup
            self.cap.release()
            self.writer.release()
            cv2.destroyAllWindows()
            
            self.logger.success(f"Processing complete! Output saved: {self.output_path}")
            detector.print_statistics()
        
        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
            raise
        
        finally:
            if self.cap:
                self.cap.release()
            if self.writer:
                self.writer.release()
            cv2.destroyAllWindows()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Advanced License Plate Detection & Recognition System'
    )
    parser.add_argument('--input', type=str, default=Config.DEFAULT_INPUT_VIDEO,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=Config.DEFAULT_OUTPUT_VIDEO,
                       help='Path to output video file')
    parser.add_argument('--model', type=str, default=Config.YOLO_MODEL_PATH,
                       help='Path to YOLO model')
    parser.add_argument('--confidence', type=float, default=Config.YOLO_CONFIDENCE,
                       help='YOLO detection confidence threshold')
    parser.add_argument('--gpu', type=bool, default=Config.USE_GPU,
                       help='Use GPU for OCR')
    parser.add_argument('--display', type=bool, default=Config.SHOW_DISPLAY,
                       help='Display output in real-time')
    parser.add_argument('--max-frames', type=int, default=Config.MAX_FRAMES,
                       help='Maximum frames to process (None for all)')
    parser.add_argument('--verbose', type=bool, default=Config.VERBOSE,
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.DEFAULT_INPUT_VIDEO = args.input
    config.DEFAULT_OUTPUT_VIDEO = args.output
    config.YOLO_MODEL_PATH = args.model
    config.YOLO_CONFIDENCE = args.confidence
    config.USE_GPU = args.gpu
    config.SHOW_DISPLAY = args.display
    config.MAX_FRAMES = args.max_frames
    config.VERBOSE = args.verbose
    
    # Initialize logger
    logger = Logger(verbose=config.VERBOSE)
    
    logger.info("="*60)
    logger.info("ADVANCED LICENSE PLATE DETECTION SYSTEM")
    logger.info("="*60)
    
    try:
        # Initialize detector
        logger.info("Initializing license plate detector...")
        detector = PlateDetector(config=config, logger=logger)
        logger.success("Detector initialized successfully")
        
        # Process video
        logger.info("Starting video processing...")
        processor = VideoProcessor(
            config.DEFAULT_INPUT_VIDEO,
            config.DEFAULT_OUTPUT_VIDEO,
            config=config,
            logger=logger
        )
        processor.process(detector)
        
        logger.success("All done! 🎉")
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
