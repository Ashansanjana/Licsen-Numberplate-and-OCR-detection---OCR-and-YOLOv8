"""
VisionAI Pro - Backend API
FastAPI server for license plate & traffic sign detection using YOLO + EasyOCR
With video processing capabilities
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import uvicorn
import os
import re
import tempfile
import uuid
from pathlib import Path

# Try to import detection libraries
try:
    from ultralytics import YOLO
    print("✅ YOLO loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import YOLO: {e}")
    YOLO = None

try:
    import easyocr
    print("✅ EasyOCR loaded successfully")
except ImportError as e:
    print(f"❌ Failed to import EasyOCR: {e}")
    easyocr = None

# Initialize FastAPI
app = FastAPI(title="VisionAI Pro API", version="3.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory for processed videos
OUTPUT_DIR = Path("./processed_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model instances
plate_model = None
traffic_sign_model = None
ocr_reader = None

# Traffic sign class names (update based on your model's training)
TRAFFIC_SIGN_CLASSES = {
    0: "Speed Limit",
    1: "Stop",
    2: "Yield",
    3: "No Entry",
    4: "Pedestrian Crossing",
    5: "Traffic Light",
    6: "Turn Right",
    7: "Turn Left",
    8: "Straight Only",
    9: "No Parking",
    10: "Warning",
    11: "School Zone",
    12: "Hospital",
    13: "Railway Crossing",
    14: "One Way"
}

def initialize_models():
    """Initialize YOLO and EasyOCR models."""
    global plate_model, traffic_sign_model, ocr_reader
    
    # Load Plate Detection YOLO model
    plate_model_path = "best.pt"
    fallback_path = "yolov8n.pt"
    
    if YOLO:
        try:
            if os.path.exists(plate_model_path):
                plate_model = YOLO(plate_model_path)
                print(f"✅ Plate detection model loaded: {plate_model_path}")
            elif os.path.exists(fallback_path):
                plate_model = YOLO(fallback_path)
                print(f"✅ Plate detection fallback model loaded: {fallback_path}")
            else:
                print("❌ No plate detection model found")
        except Exception as e:
            print(f"❌ Plate model initialization error: {e}")
        
        # Load Traffic Sign Detection YOLO model
        traffic_sign_path = "traffic_sign.pt"
        try:
            if os.path.exists(traffic_sign_path):
                traffic_sign_model = YOLO(traffic_sign_path)
                print(f"✅ Traffic sign model loaded: {traffic_sign_path}")
            else:
                print(f"⚠️ Traffic sign model not found: {traffic_sign_path}")
        except Exception as e:
            print(f"❌ Traffic sign model initialization error: {e}")
    
    # Load EasyOCR reader
    if easyocr:
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=True)
            print("✅ EasyOCR initialized (GPU)")
        except Exception as e:
            try:
                ocr_reader = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR initialized (CPU)")
            except Exception as e2:
                print(f"❌ EasyOCR initialization error: {e2}")

# Initialize on startup
initialize_models()

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    if image is None or image.size == 0:
        return ""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def clean_plate_text(text):
    """Clean OCR output text."""
    if not text:
        return ""
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9\-\s]', '', text)
    text = re.sub(r'[\s]+', ' ', text)
    return text.strip() if len(text) >= 2 else ""

def read_plate_text(crop):
    """Read plate text using EasyOCR - no preprocessing."""
    if ocr_reader is None or crop is None:
        return "", 0.0
    
    try:
        results = ocr_reader.readtext(crop)
        
        if not results:
            return "", 0.0
        
        full_text = ""
        total_conf = 0.0
        count = 0
        
        for detection in results:
            text = detection[1]
            conf = detection[2]
            if conf > 0.2:
                full_text += " " + text
                total_conf += conf
                count += 1
        
        avg_conf = total_conf / count if count > 0 else 0.0
        cleaned = clean_plate_text(full_text)
        
        return cleaned, avg_conf
        
    except Exception as e:
        print(f"OCR error: {e}")
        return "", 0.0

def detect_plates_in_image(img):
    """Detect license plates in an image."""
    if plate_model is None:
        return []
    
    h, w = img.shape[:2]
    results = plate_model(img, conf=0.4, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            det_conf = float(box.conf[0].cpu().numpy())
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            if box_w < 20 or box_h < 10 or box_w > w * 0.8 or box_h > h * 0.5:
                continue
            
            # Crop plate with padding
            pad = 5
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            
            crop = img[cy1:cy2, cx1:cx2]
            plate_text, ocr_conf = read_plate_text(crop)
            crop_b64 = encode_image_to_base64(crop)
            
            detections.append({
                "id": len(detections),
                "type": "plate",
                "bbox": [x1, y1, box_w, box_h],
                "text": plate_text if plate_text else "UNREADABLE",
                "confidence": round(ocr_conf, 2),
                "detection_conf": round(det_conf, 2),
                "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None
            })
    
    return detections

def detect_traffic_signs_in_image(img):
    """Detect traffic signs in an image."""
    if traffic_sign_model is None:
        return []
    
    h, w = img.shape[:2]
    results = traffic_sign_model(img, conf=0.4, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            det_conf = float(box.conf[0].cpu().numpy())
            
            # Get class ID and name
            class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
            class_name = TRAFFIC_SIGN_CLASSES.get(class_id, f"Sign {class_id}")
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            if box_w < 15 or box_h < 15:
                continue
            
            # Crop sign
            pad = 3
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            
            crop = img[cy1:cy2, cx1:cx2]
            crop_b64 = encode_image_to_base64(crop)
            
            detections.append({
                "id": len(detections),
                "type": "traffic_sign",
                "bbox": [x1, y1, box_w, box_h],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(det_conf, 2),
                "crop_image": f"data:image/jpeg;base64,{crop_b64}" if crop_b64 else None
            })
    
    return detections

def draw_detections_on_frame(frame, plates, signs):
    """Draw detection boxes on a frame for video output."""
    # Draw plate detections (cyan)
    for det in plates:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan in BGR
        
        # Draw label
        label = det.get("text", "PLATE")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Draw traffic sign detections (orange)
    for det in signs:
        x, y, w, h = det["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)  # Orange in BGR
        
        # Draw label
        label = det.get("class_name", "SIGN")
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return frame

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "VisionAI Pro API v3.0",
        "plate_model_loaded": plate_model is not None,
        "traffic_sign_model_loaded": traffic_sign_model is not None,
        "ocr_loaded": ocr_reader is not None
    }

@app.post("/detect")
async def detect_all(file: UploadFile = File(...)):
    """
    Detect both license plates and traffic signs in uploaded image.
    Returns bounding boxes, extracted text, confidence scores, and cropped images.
    """
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect plates
        plates = detect_plates_in_image(img)
        
        # Detect traffic signs
        signs = detect_traffic_signs_in_image(img)
        
        return JSONResponse(content={
            "detections": plates,  # Keep backward compatibility
            "plates": plates,
            "traffic_signs": signs,
            "total_plates": len(plates),
            "total_signs": len(signs)
        })
    
    except Exception as e:
        print(f"Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Process video file for license plate and traffic sign detection.
    Returns processed video with detection annotations.
    """
    if plate_model is None and traffic_sign_model is None:
        raise HTTPException(status_code=503, detail="No detection models initialized")
    
    try:
        # Save uploaded video to temp file
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input.write(await file.read())
        temp_input.close()
        
        # Open video
        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            os.unlink(temp_input.name)
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output file
        output_id = str(uuid.uuid4())[:8]
        output_filename = f"detected_{output_id}.mp4"
        output_path = OUTPUT_DIR / output_filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        all_plates = []
        all_signs = []
        
        # Process every 2nd frame for speed, but write all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect on every 3rd frame for performance
            if frame_count % 3 == 0:
                plates = detect_plates_in_image(frame)
                signs = detect_traffic_signs_in_image(frame)
                
                # Track unique detections
                for p in plates:
                    p["frame"] = frame_count
                    all_plates.append(p)
                for s in signs:
                    s["frame"] = frame_count
                    all_signs.append(s)
            else:
                plates = []
                signs = []
            
            # Draw detections
            annotated = draw_detections_on_frame(frame.copy(), plates, signs)
            out.write(annotated)
        
        cap.release()
        out.release()
        os.unlink(temp_input.name)
        
        # Get unique plate texts
        unique_plates = list(set(p["text"] for p in all_plates if p["text"] != "UNREADABLE"))
        unique_signs = list(set(s["class_name"] for s in all_signs))
        
        return JSONResponse(content={
            "success": True,
            "output_file": output_filename,
            "download_url": f"/download/{output_filename}",
            "stats": {
                "total_frames": total_frames,
                "processed_frames": frame_count,
                "plates_detected": len(all_plates),
                "signs_detected": len(all_signs),
                "unique_plates": unique_plates,
                "unique_signs": unique_signs
            }
        })
    
    except Exception as e:
        print(f"Video processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download processed video file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
