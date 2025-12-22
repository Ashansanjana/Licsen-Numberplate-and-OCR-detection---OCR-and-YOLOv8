"""
Number Plate Detection System
- YOLOv8 for vehicle and plate detection
- ByteTrack for vehicle tracking
- Tesseract OCR for plate text recognition
- Enhanced preprocessing for improved accuracy
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from util import read_license_plate, enhance_plate_image, validate_crop_quality

# ============================================================================
# CONFIGURATION
# ============================================================================
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "best.pt"
VIDEO_INPUT_PATH = "sample_video.mp4"
VIDEO_OUTPUT_PATH = "output_video.mp4"

VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
PLATE_CONF_THRESHOLD = 0.4  # Minimum confidence for plate detection
MIN_PLATE_WIDTH = 20  # Minimum plate width in pixels
MIN_PLATE_HEIGHT = 10  # Minimum plate height in pixels

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_models():
    """Load YOLO models for vehicle and plate detection."""
    print("📥 Loading YOLO models...")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    print("✅ Models loaded successfully")
    return vehicle_model, plate_model


def initialize_video_capture(video_path):
    """Initialize video capture from file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"📹 Video: {width}x{height} @ {fps} FPS")
    return cap, width, height, fps


def initialize_video_writer(output_path, width, height, fps):
    """Initialize video writer for output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"❌ Cannot create video writer: {output_path}")
    
    print(f"🎬 Output video: {output_path}")
    return out


# ============================================================================
# DETECTION & TRACKING
# ============================================================================

def detect_vehicles(frame, vehicle_model):
    """Detect vehicles in frame using YOLOv8."""
    results = vehicle_model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # Filter by vehicle classes
    detections = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]
    return detections


def detect_plates(frame_original, plate_model):
    """Detect license plates in frame."""
    results = plate_model(frame_original, verbose=False, conf=PLATE_CONF_THRESHOLD)[0]
    return results.boxes.data.tolist()  # Returns: [x1, y1, x2, y2, score, cls]


def crop_plate(frame, x1, y1, x2, y2, pad_pixels=5):
    """
    Crop plate region from frame with optional padding.
    
    Args:
        frame: Input image
        x1, y1, x2, y2: Bounding box coordinates
        pad_pixels: Padding around the crop
        
    Returns:
        Cropped image or None if invalid
    """
    height, width = frame.shape[:2]
    
    # Apply padding
    y1 = max(0, int(y1) - pad_pixels)
    x1 = max(0, int(x1) - pad_pixels)
    y2 = min(height, int(y2) + pad_pixels)
    x2 = min(width, int(x2) + pad_pixels)
    
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Validate crop dimensions
    if crop_width < MIN_PLATE_WIDTH or crop_height < MIN_PLATE_HEIGHT:
        return None
    
    crop = frame[y1:y2, x1:x2]
    
    # Verify crop is not empty or corrupted
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    
    return crop


def resize_plate_if_small(crop, target_width=200):
    """
    Resize small plates to improve OCR accuracy.
    
    Args:
        crop: Plate image
        target_width: Target width for small plates
        
    Returns:
        Resized or original image
    """
    height, width = crop.shape[:2]
    
    # If plate is too small, resize it for better OCR
    if width < target_width:
        scale_factor = target_width / width
        new_height = int(height * scale_factor)
        crop = cv2.resize(crop, (target_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return crop


def draw_plate_detection(frame, x1, y1, x2, y2, text, score, ocr_conf):
    """Draw enhanced plate bounding box and OCR text on frame."""
    # Draw main bounding box with gradient effect
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    
    # Draw corner markers for better visibility
    corner_length = 20
    thickness = 2
    color = (0, 255, 0)
    
    # Top-left corner
    cv2.line(frame, (int(x1), int(y1)), (int(x1) + corner_length, int(y1)), color, thickness)
    cv2.line(frame, (int(x1), int(y1)), (int(x1), int(y1) + corner_length), color, thickness)
    
    # Top-right corner
    cv2.line(frame, (int(x2), int(y1)), (int(x2) - corner_length, int(y1)), color, thickness)
    cv2.line(frame, (int(x2), int(y1)), (int(x2), int(y1) + corner_length), color, thickness)
    
    # Bottom-left corner
    cv2.line(frame, (int(x1), int(y2)), (int(x1) + corner_length, int(y2)), color, thickness)
    cv2.line(frame, (int(x1), int(y2)), (int(x1), int(y2) - corner_length), color, thickness)
    
    # Bottom-right corner
    cv2.line(frame, (int(x2), int(y2)), (int(x2) - corner_length, int(y2)), color, thickness)
    cv2.line(frame, (int(x2), int(y2)), (int(x2), int(y2) - corner_length), color, thickness)
    
    # Draw main text with background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
    text_y = int(y1) - 15
    text_x = int(x1)
    
    # Background for main text
    cv2.rectangle(frame, 
                  (text_x - 5, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 5, text_y + 5),
                  (0, 255, 0), -1)
    
    # Main text (plate number)
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    
    # Draw confidence info below
    conf_text = f"Det:{score:.2f} OCR:{ocr_conf:.2f}"
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    
    cv2.rectangle(frame,
                  (int(x1), int(y2) + 5),
                  (int(x1) + conf_size[0] + 10, int(y2) + conf_size[1] + 15),
                  (100, 100, 255), -1)
    
    cv2.putText(frame, conf_text, (int(x1) + 5, int(y2) + conf_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main detection and tracking pipeline."""
    # Initialize models
    vehicle_model, plate_model = initialize_models()
    
    # Initialize video
    cap, width, height, fps = initialize_video_capture(VIDEO_INPUT_PATH)
    out = initialize_video_writer(VIDEO_OUTPUT_PATH, width, height, fps)
    
    # Initialize tracker
    tracker = sv.ByteTrack()
    box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.7)
    
    frame_count = 0
    plate_detections = []
    
    print("🚗 Starting detection pipeline...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ End of video reached")
                break
            
            frame_count += 1
            original_frame = frame.copy()  # Keep clean copy for OCR
            
            # ================================================================
            # VEHICLE DETECTION & TRACKING
            # ================================================================
            detections = detect_vehicles(original_frame, vehicle_model)
            tracked_vehicles = tracker.update_with_detections(detections)
            
            # Generate labels
            vehicle_labels = [f"ID{track_id}" for track_id in tracked_vehicles.tracker_id]
            
            # Annotate vehicles on display frame (NOT for plate detection)
            frame = box_annotator.annotate(frame, tracked_vehicles)
            frame = label_annotator.annotate(frame, tracked_vehicles, vehicle_labels)
            
            # ================================================================
            # PLATE DETECTION (run on original clean frame)
            # ================================================================
            plate_boxes = detect_plates(original_frame, plate_model)
            
            for box in plate_boxes:
                x1, y1, x2, y2, score, cls = box
                
                # Crop plate from ORIGINAL frame (before annotation)
                crop = crop_plate(original_frame, x1, y1, x2, y2, pad_pixels=5)
                
                if crop is None:
                    continue
                
                # Validate crop quality
                if not validate_crop_quality(crop):
                    continue
                
                # Resize if too small
                crop = resize_plate_if_small(crop, target_width=200)
                
                # Enhance plate image
                crop_enhanced = enhance_plate_image(crop)
                
                # Read license plate with OCR
                text, ocr_conf = read_license_plate(crop_enhanced)
                
                if text:
                    # Draw on display frame (enhanced visualization)
                    draw_plate_detection(frame, x1, y1, x2, y2, text, score, ocr_conf)
                    
                    # Log detection
                    plate_detections.append({
                        'frame': frame_count,
                        'plate_text': text,
                        'detection_confidence': score,
                        'ocr_confidence': ocr_conf
                    })
                    
                    print(f"Frame {frame_count}: [{text}] (Det: {score:.2f}, OCR: {ocr_conf:.2f})")
            
            # Write annotated frame to output
            out.write(frame)
            
            # Display live (optional - comment out for headless mode)
            cv2.imshow("License Plate Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⏹️  User stopped processing")
                break
            
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "="*60)
        print(f"📊 PROCESSING COMPLETE")
        print(f"   Total frames: {frame_count}")
        print(f"   Plates detected: {len(plate_detections)}")
        print(f"   Output saved: {VIDEO_OUTPUT_PATH}")
        print("="*60 + "\n")
        
        return plate_detections


if __name__ == "__main__":
    plate_detections = main()
