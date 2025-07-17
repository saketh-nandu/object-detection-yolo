import cv2
import torch
import numpy as np
import time
from datetime import datetime
import threading
import json
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YOLOSecuritySystem:
    def __init__(self, model_path: str = 'yolov5s.pt', conf_threshold: float = 0.5):
        """
        Initialize YOLO Security Surveillance System
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        self.model = self._load_model()
        
        # Security-specific settings
        self.security_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'bus': 5,
            'truck': 7, 'knife': 43, 'cell phone': 67, 'laptop': 63
        }
        
        # Alert system
        self.alert_history = deque(maxlen=100)
        self.last_alert_time = {}
        self.alert_cooldown = 30  # seconds
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'alerts_sent': 0,
            'uptime_start': datetime.now(),
            'fps': 0
        }
        
        # Frame processing
        self.frame_queue = deque(maxlen=30)
        self.is_recording = False
        self.recording_buffer = deque(maxlen=300)  # 10 seconds at 30fps
        
        logger.info(f"YOLO Security System initialized on {self.device}")
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            # Try to load YOLOv5 from torch hub
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.to(self.device)
            model.conf = self.conf_threshold
            logger.info("YOLOv5 model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Perform object detection on a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        try:
            # Run inference
            results = self.model(frame)
            
            # Parse results
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = self.model.names[int(cls)]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': class_name,
                        'class_id': int(cls),
                        'timestamp': datetime.now().isoformat()
                    }
                    detections.append(detection)
            
            self.detection_stats['total_detections'] += len(detections)
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def is_security_threat(self, detections: List[Dict]) -> bool:
        """
        Analyze detections for security threats
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            True if threat detected
        """
        threat_classes = ['person', 'knife', 'car', 'motorcycle', 'truck']
        
        for detection in detections:
            if detection['class'] in threat_classes:
                # Additional threat logic
                if detection['class'] == 'person' and detection['confidence'] > 0.7:
                    return True
                elif detection['class'] == 'knife' and detection['confidence'] > 0.6:
                    return True
                elif detection['class'] in ['car', 'motorcycle', 'truck'] and detection['confidence'] > 0.8:
                    return True
        
        return False

    def send_alert(self, detections: List[Dict], frame: np.ndarray):
        """
        Send security alert with detection information
        
        Args:
            detections: List of detections that triggered alert
            frame: Current frame for evidence
        """
        current_time = datetime.now()
        alert_key = f"{current_time.strftime('%Y%m%d_%H%M')}"
        
        # Check cooldown
        if alert_key in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[alert_key]).seconds
            if time_diff < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = current_time
        
        # Create alert record
        alert_record = {
            'timestamp': current_time.isoformat(),
            'detections': detections,
            'threat_level': self._assess_threat_level(detections),
            'location': 'Camera_1',  # Configurable
            'alert_id': len(self.alert_history) + 1
        }
        
        self.alert_history.append(alert_record)
        self.detection_stats['alerts_sent'] += 1
        
        # Log alert
        threat_objects = [d['class'] for d in detections]
        logger.warning(f"SECURITY ALERT: {threat_objects} detected at {current_time}")
        
        # Save evidence frame
        self._save_evidence_frame(frame, alert_record['alert_id'])
        
        # Send notification (implement email/SMS as needed)
        self._send_notification(alert_record)

    def _assess_threat_level(self, detections: List[Dict]) -> str:
        """Assess threat level based on detections"""
        max_threat = "LOW"
        
        for detection in detections:
            if detection['class'] == 'knife':
                max_threat = "HIGH"
            elif detection['class'] == 'person' and detection['confidence'] > 0.8:
                max_threat = "MEDIUM" if max_threat == "LOW" else max_threat
            elif detection['class'] in ['car', 'motorcycle', 'truck']:
                max_threat = "MEDIUM" if max_threat == "LOW" else max_threat
        
        return max_threat

    def _save_evidence_frame(self, frame: np.ndarray, alert_id: int):
        """Save frame as evidence"""
        try:
            evidence_dir = Path("evidence")
            evidence_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = evidence_dir / f"alert_{alert_id}_{timestamp}.jpg"
            
            cv2.imwrite(str(filename), frame)
            logger.info(f"Evidence saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")

    def _send_notification(self, alert_record: Dict):
        """Send notification (placeholder for email/SMS integration)"""
        # Implement actual notification system here
        print(f"🚨 ALERT: {alert_record['threat_level']} threat detected!")
        print(f"   Objects: {[d['class'] for d in alert_record['detections']]}")
        print(f"   Time: {alert_record['timestamp']}")
        print(f"   Alert ID: {alert_record['alert_id']}")

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Color coding for security classes
            if class_name in ['person', 'knife']:
                color = (0, 0, 255)  # Red for high priority
            elif class_name in ['car', 'motorcycle', 'truck']:
                color = (0, 255, 255)  # Yellow for medium priority
            else:
                color = (0, 255, 0)  # Green for normal objects
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame

    def process_video_stream(self, source: int = 0, display: bool = True):
        """
        Process live video stream from camera
        
        Args:
            source: Camera source (0 for default webcam)
            display: Whether to display video window
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera source: {source}")
            return
        
        logger.info(f"Starting video stream from source: {source}")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Perform detection
                detections = self.detect_objects(frame)
                
                # Check for security threats
                if self.is_security_threat(detections):
                    self.send_alert(detections, frame)
                    
                    # Start recording if not already
                    if not self.is_recording:
                        self.is_recording = True
                        threading.Thread(target=self._record_incident, 
                                       args=(frame.copy(),), daemon=True).start()
                
                # Draw detections
                if display:
                    annotated_frame = self.draw_detections(frame, detections)
                    
                    # Add system info overlay
                    self._add_system_overlay(annotated_frame)
                    
                    cv2.imshow('YOLO Security System', annotated_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = time.time()
                    self.detection_stats['fps'] = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                # Add to frame queue
                self.frame_queue.append(frame.copy())
                
        except KeyboardInterrupt:
            logger.info("Stopping video stream...")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info("Video stream stopped")

    def _add_system_overlay(self, frame: np.ndarray):
        """Add system information overlay to frame"""
        # System status
        uptime = datetime.now() - self.detection_stats['uptime_start']
        uptime_str = str(uptime).split('.')[0]
        
        overlay_text = [
            f"Status: ACTIVE",
            f"FPS: {self.detection_stats['fps']:.1f}",
            f"Detections: {self.detection_stats['total_detections']}",
            f"Alerts: {self.detection_stats['alerts_sent']}",
            f"Uptime: {uptime_str}",
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        y_offset = 30
        for text in overlay_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            y_offset += 25

    def _record_incident(self, trigger_frame: np.ndarray):
        """Record incident video clip"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"incident_{timestamp}.mp4"
            
            # Video writer setup
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, 
                                 (trigger_frame.shape[1], trigger_frame.shape[0]))
            
            # Record for 10 seconds
            record_time = time.time()
            while time.time() - record_time < 10:
                if self.frame_queue:
                    frame = self.frame_queue[-1]
                    out.write(frame)
                time.sleep(1/30)  # 30 FPS
            
            out.release()
            self.is_recording = False
            logger.info(f"Incident recorded: {filename}")
            
        except Exception as e:
            logger.error(f"Recording error: {e}")

    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        uptime = datetime.now() - self.detection_stats['uptime_start']
        
        return {
            'status': 'ACTIVE',
            'uptime': str(uptime).split('.')[0],
            'total_detections': self.detection_stats['total_detections'],
            'alerts_sent': self.detection_stats['alerts_sent'],
            'current_fps': self.detection_stats['fps'],
            'recent_alerts': list(self.alert_history)[-5:],  # Last 5 alerts
            'device': str(self.device),
            'model_path': self.model_path
        }

    def save_config(self, config_path: str = "security_config.json"):
        """Save current configuration"""
        config = {
            'model_path': self.model_path,
            'conf_threshold': self.conf_threshold,
            'alert_cooldown': self.alert_cooldown,
            'security_classes': self.security_classes
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")

# Example usage and testing
def main():
    """Main function to run the security system"""
    
    # Initialize system
    security_system = YOLOSecuritySystem(
        model_path='yolov5s.pt',
        conf_threshold=0.5
    )
    
    # Save configuration
    security_system.save_config()
    
    print("🔒 YOLO Security Surveillance System")
    print("=" * 40)
    print("Features:")
    print("- Real-time object detection")
    print("- Security threat assessment")
    print("- Automatic alert system")
    print("- Evidence recording")
    print("- Performance monitoring")
    print("\nPress 'q' to quit")
    print("=" * 40)
    
    try:
        # Start video processing
        security_system.process_video_stream(source=0, display=True)
        
    except Exception as e:
        logger.error(f"System error: {e}")
    
    finally:
        # Print final statistics
        stats = security_system.get_system_stats()
        print("\n📊 Final Statistics:")
        print(f"Uptime: {stats['uptime']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Alerts Sent: {stats['alerts_sent']}")
        print(f"Average FPS: {stats['current_fps']:.1f}")

if __name__ == "__main__":
    main()