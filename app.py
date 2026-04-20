import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
import os
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import threading
import time
import queue

app = Flask(__name__)
app.config['SECRET_KEY'] = 'theater_phone_detector_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///theater_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Global camera manager
camera_manager = None
frame_queue = queue.Queue(maxsize=5)
detection_active = False

class CameraManager:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.detection_enabled = False
        self.current_screen_id = None  # Store screen ID for detection saving
        self.last_detection_time = 0  # Prevent spam detection saving
        
    def start_camera(self):
        """Start camera in a separate thread for immediate response"""
        if self.running:
            return True
            
        print("Initializing camera...")
        # Try multiple backends for best compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(0, backend)
                if self.cap.isOpened():
                    # Optimize camera settings
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Test capture
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"Camera initialized successfully with backend: {backend}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"Failed to initialize with backend {backend}: {e}")
                continue
        
        if self.cap is None or not self.cap.isOpened():
            print("Failed to initialize camera with any backend")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Camera thread started")
        return True
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        frame_count = 0
        last_log_time = time.time()
        
        while self.running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame_count += 1
                    
                    # Log progress every 30 frames
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:  # Log every 5 seconds
                        print(f"Camera capturing: Frame {frame_count}, Queue size: {frame_queue.qsize()}")
                        last_log_time = current_time
                    
                    # Resize frame for better performance
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Process frame if detection is enabled
                    if self.detection_enabled:
                        try:
                            frame = self._process_frame(frame, frame_count)
                        except Exception as e:
                            print(f"⚠️  Detection processing error (continuing): {e}")
                            # Add basic overlay even if detection fails
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            cv2.putText(frame, f"Frame: {frame_count} | {timestamp}", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, "DETECTION ERROR - MONITORING CONTINUES", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    else:
                        # Add basic overlay even without detection
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, f"Frame: {frame_count} | {timestamp}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, "CAMERA READY", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Store current frame with thread safety
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        
                    # Add to queue for streaming (non-blocking)
                    try:
                        frame_queue.put(frame, block=False)
                    except queue.Full:
                        # Remove old frame and add new one
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put(frame, block=False)
                        except queue.Empty:
                            pass
                else:
                    print("Failed to capture frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in capture loop: {e}")
                time.sleep(0.1)
        
        print(f"Capture loop ended after {frame_count} frames")
    
    def _process_frame(self, frame, frame_count):
        """Process frame with YOLO detection"""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Frame: {frame_count} | {timestamp}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add status overlay
            cv2.putText(frame, "MONITORING ACTIVE", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # YOLO detection
            global yolo_model
            if yolo_model:
                results = yolo_model(frame, conf=0.3, verbose=False)
                
                # Draw bounding boxes for phone detections
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class and confidence
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            
                            # Check if it's a phone (class 67) with good confidence
                            if cls == 67 and conf > 0.3:  # 'cell phone' class
                                # Get coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                # Add label
                                label = f"PHONE DETECTED! {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                
                                print(f"📱 Phone detected with confidence: {conf:.2f}")
                                
                                # Save detection immediately (with spam protection built-in)
                                # Use try-catch to ensure detection errors don't stop video
                                try:
                                    self._save_detection(frame, x1, y1, x2, y2, conf)
                                except Exception as e:
                                    print(f"⚠️  Detection save error (video continues): {e}")
                                    # Don't re-raise - keep video running
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return frame
    
    def _save_detection(self, frame, x1, y1, x2, y2, confidence):
        """Save detection to database"""
        try:
            # Check if we have a screen ID and prevent spam
            if not self.current_screen_id:
                print("No screen ID available for detection saving")
                return
            
            # Prevent spam - only save once every 5 seconds
            current_time = time.time()
            if current_time - self.last_detection_time < 5.0:
                return
            
            self.last_detection_time = current_time
            
            # Use application context for all database operations
            with app.app_context():
                # Get screen info
                screen = Screen.query.get(self.current_screen_id)
                if not screen:
                    print(f"Screen {self.current_screen_id} not found")
                    return
                
                # Generate random seat number for demo
                import random
                seat_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
                seat_number = random.choice(seat_letters) + str(random.randint(1, 20))
                
                # Create evidence directory
                evidence_dir = 'static/evidence'
                os.makedirs(evidence_dir, exist_ok=True)
                
                # Create filename with timestamp and detection info
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                evidence_filename = f"detection_{timestamp}_{seat_number}_conf{confidence:.2f}.jpg"
                evidence_path = os.path.join(evidence_dir, evidence_filename)
                
                # Create a copy of the frame with highlighted detection
                detection_frame = frame.copy()
                
                # Draw enhanced bounding box for evidence
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # Add detection info overlay
                cv2.putText(detection_frame, f"PHONE DETECTED!", (x1, y1-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(detection_frame, f"Confidence: {confidence:.2f}", (x1, y1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(detection_frame, f"Seat: {seat_number}", (x1, y2+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(detection_frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (x1, y2+50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Save the annotated frame as evidence
                cv2.imwrite(evidence_path, detection_frame)
                
                # Create detection record in database
                detection = Detection(
                    screen_id=self.current_screen_id,
                    detection_time=datetime.now(),
                    confidence=confidence,
                    seat_number=seat_number,
                    evidence_path=evidence_filename,
                    x_coordinate=x1,
                    y_coordinate=y1
                )
                
                db.session.add(detection)
                db.session.commit()
                
                print(f"✅ Detection saved: {seat_number} at {timestamp} with confidence {confidence:.2f}")
                print(f"Evidence photo: {evidence_filename}")
            
        except Exception as e:
            print(f"❌ Error saving detection: {e}")
            import traceback
            traceback.print_exc()
    
    def enable_detection(self, enable=True, screen_id=None):
        """Enable/disable phone detection"""
        self.detection_enabled = enable
        if screen_id:
            self.current_screen_id = screen_id
        global detection_active
        detection_active = enable
        print(f"Detection {'enabled' if enable else 'disabled'} for screen {screen_id}")
    
    def get_current_frame(self):
        """Get the latest frame safely"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def stop_camera(self):
        """Stop camera and cleanup"""
        print("Stopping camera...")
        self.running = False
        self.detection_enabled = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("Camera stopped")

# Global variables for camera and detection
camera = None
detection_active = False
yolo_model = None

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    theater_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Screen(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    total_seats = db.Column(db.Integer, nullable=False)
    rows = db.Column(db.Integer, nullable=False)
    seats_per_row = db.Column(db.Integer, nullable=False)
    theater_layout = db.Column(db.Text)  # JSON string for seat layout
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    screen_id = db.Column(db.Integer, db.ForeignKey('screen.id'), nullable=False)
    seat_number = db.Column(db.String(10), nullable=False)
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    confidence = db.Column(db.Float, nullable=False)
    evidence_path = db.Column(db.String(200), nullable=False)
    x_coordinate = db.Column(db.Integer, nullable=False)
    y_coordinate = db.Column(db.Integer, nullable=False)
    resolved = db.Column(db.Boolean, default=False)

# Initialize YOLO model
def init_yolo_model():
    global yolo_model
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['theater_name'] = user.theater_name
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    screens = Screen.query.filter_by(user_id=user_id).all()
    
    # Get recent detections
    recent_detections = db.session.query(Detection, Screen).join(Screen).filter(
        Screen.user_id == user_id
    ).order_by(Detection.detection_time.desc()).limit(10).all()
    
    return render_template('dashboard.html', screens=screens, recent_detections=recent_detections)

@app.route('/screen/<int:screen_id>')
def screen_monitor(screen_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    screen = Screen.query.get_or_404(screen_id)
    if screen.user_id != session['user_id']:
        return redirect(url_for('dashboard'))
    
    return render_template('screen_monitor.html', screen=screen)

@app.route('/start_detection/<int:screen_id>')
def start_detection(screen_id):
    global detection_active, camera_manager
    
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    screen = Screen.query.get_or_404(screen_id)
    if screen.user_id != session['user_id']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        if not camera_manager:
            return jsonify({'error': 'Camera manager not initialized'}), 500
        
        # Start camera if not already running
        if not camera_manager.running:
            success = camera_manager.start_camera()
            if not success:
                return jsonify({'error': 'Failed to initialize camera. Please check if camera is connected and not in use by another application.'}), 500
        
        # Enable detection
        camera_manager.enable_detection(True, screen_id)
        detection_active = True
        session['current_screen_id'] = screen_id
        
        print(f"Detection started for screen {screen_id} - Camera ready")
        return jsonify({
            'success': True, 
            'message': 'Detection started successfully',
            'camera_status': 'Camera initialized and running'
        })
        
    except Exception as e:
        print(f"Error starting detection: {e}")
        return jsonify({'error': f'Failed to start detection: {str(e)}'}), 500

@app.route('/stop_detection')
def stop_detection():
    global detection_active, camera_manager
    
    detection_active = False
    
    if camera_manager:
        camera_manager.enable_detection(False)
    
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/detection_stats/<int:screen_id>')
def detection_stats(screen_id):
    """Get detection statistics for validation"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        # Get total detections for this screen
        total_detections = Detection.query.filter_by(screen_id=screen_id).count()
        
        # Get recent detections (last 10 minutes)
        recent_time = datetime.now() - timedelta(minutes=10)
        recent_detections = Detection.query.filter(
            Detection.screen_id == screen_id,
            Detection.detection_time >= recent_time
        ).count()
        
        # Get latest detection
        latest_detection = Detection.query.filter_by(screen_id=screen_id).order_by(
            Detection.detection_time.desc()
        ).first()
        
        # Check camera status
        camera_status = {
            'running': camera_manager.running if camera_manager else False,
            'detection_enabled': camera_manager.detection_enabled if camera_manager else False,
            'queue_size': frame_queue.qsize() if frame_queue else 0
        }
        
        stats = {
            'total_detections': total_detections,
            'recent_detections': recent_detections,
            'latest_detection': {
                'time': latest_detection.detection_time.strftime('%Y-%m-%d %H:%M:%S') if latest_detection else None,
                'seat': latest_detection.seat_number if latest_detection else None,
                'confidence': latest_detection.confidence if latest_detection else None
            } if latest_detection else None,
            'camera_status': camera_status,
            'validation': {
                'camera_running': camera_status['running'],
                'detection_active': camera_status['detection_enabled'],
                'has_detections': total_detections > 0,
                'system_working': camera_status['running'] and camera_status['detection_enabled']
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evidence/<filename>')
def view_evidence(filename):
    """Serve evidence photos"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    evidence_dir = 'static/evidence'
    return send_file(os.path.join(evidence_dir, filename), as_attachment=False)

@app.route('/test_detection/<int:screen_id>')
def test_detection(screen_id):
    """Create a test detection to verify the saving process"""
    try:
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "TEST DETECTION", (200, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(test_frame, f"Screen {screen_id}", (200, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw a fake bounding box
        x1, y1, x2, y2 = 200, 150, 350, 280
        cv2.rectangle(test_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Generate random seat number for demo
        import random
        seat_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        seat_number = random.choice(seat_letters) + str(random.randint(1, 20))
        confidence = 0.85
        
        # Create evidence directory
        evidence_dir = 'static/evidence'
        os.makedirs(evidence_dir, exist_ok=True)
        
        # Create filename with timestamp and detection info
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        evidence_filename = f"test_detection_{timestamp}_{seat_number}_conf{confidence:.2f}.jpg"
        evidence_path = os.path.join(evidence_dir, evidence_filename)
        
        # Save the test frame
        cv2.imwrite(evidence_path, test_frame)
        
        # Create detection record in database
        detection = Detection(
            screen_id=screen_id,
            detection_time=datetime.now(),
            confidence=confidence,
            seat_number=seat_number,
            evidence_path=evidence_filename,
            x_coordinate=x1,
            y_coordinate=y1
        )
        
        db.session.add(detection)
        db.session.commit()
        
        print(f"✅ Test detection saved: {seat_number} with confidence {confidence:.2f}")
        
        return jsonify({
            'success': True,
            'message': f'Test detection created for seat {seat_number}',
            'evidence_file': evidence_filename,
            'confidence': confidence,
            'seat_number': seat_number
        })
        
    except Exception as e:
        print(f"❌ Error creating test detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video_test')
def video_test():
    """Test page for video feed debugging"""
    return render_template('video_test.html')

@app.route('/camera_status')
def camera_status():
    """Check camera status and queue info"""
    global camera_manager
    
    status = {
        'camera_manager_exists': camera_manager is not None,
        'camera_running': camera_manager.running if camera_manager else False,
        'detection_enabled': camera_manager.detection_enabled if camera_manager else False,
        'queue_size': frame_queue.qsize(),
        'queue_full': frame_queue.full(),
        'current_frame_available': camera_manager.get_current_frame() is not None if camera_manager else False
    }
    
    return jsonify(status)

@app.route('/test_video_feed')
def test_video_feed():
    """Simple test route to check if video feed is working"""
    def generate():
        print("Test video feed - generating simple frames...")
        for i in range(10):
            # Create a simple test frame
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some text
            cv2.putText(test_frame, f"Test Frame {i+1}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', test_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.5)  # 2 FPS for testing
        print("Test video feed completed")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    def generate():
        global camera_manager
        
        print("Starting new video feed generation...")
        frame_count = 0
        
        # Check if camera manager is available
        if not camera_manager:
            print("Error: Camera manager not available - reinitializing...")
            camera_manager = CameraManager()
        
        # Auto-start camera if not running
        if not camera_manager.running:
            print("Camera not running - attempting to start...")
            success = camera_manager.start_camera()
            if not success:
                print("Failed to start camera - creating fallback frames")
                # Generate fallback frames
                for i in range(100):
                    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(fallback_frame, "Camera Unavailable", (150, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(fallback_frame, "Please check camera connection", (100, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(fallback_frame, f"Frame {i+1}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', fallback_frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n'
                               b'\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.1)
                return
        
        # Wait for camera to be ready and producing frames
        wait_count = 0
        while wait_count < 30 and frame_queue.empty():
            time.sleep(0.1)
            wait_count += 1
            if wait_count % 10 == 0:
                print(f"Waiting for camera frames... ({wait_count/10} seconds)")
        
        print("Video feed ready, starting stream...")
        
        # Track consecutive failures
        consecutive_failures = 0
        max_failures = 50
        
        while camera_manager and camera_manager.running and consecutive_failures < max_failures:
            try:
                # Get frame from queue (non-blocking with timeout)
                frame = None
                try:
                    frame = frame_queue.get(timeout=1.0)
                    consecutive_failures = 0  # Reset on success
                    frame_count += 1
                    
                    if frame_count % 30 == 0:
                        print(f"Video feed: Streamed {frame_count} frames, Queue size: {frame_queue.qsize()}")
                        
                except queue.Empty:
                    consecutive_failures += 1
                    if consecutive_failures % 10 == 0:
                        print(f"Warning: No frames available ({consecutive_failures} consecutive failures)")
                    
                    # Try to get current frame directly as fallback
                    frame = camera_manager.get_current_frame()
                    if frame is None:
                        continue
                
                if frame is None:
                    consecutive_failures += 1
                    continue
                
                # Encode frame to JPEG with optimized settings
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
                ret, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if not ret:
                    print("Failed to encode frame")
                    consecutive_failures += 1
                    continue
                    
                frame_bytes = buffer.tobytes()
                
                # Yield frame in proper MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + f"{len(frame_bytes)}".encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"Error in video feed generation: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                
        print(f"Video feed ended after {frame_count} frames (consecutive failures: {consecutive_failures})")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def save_detection(frame, x1, y1, x2, y2, confidence):
    """Save detection to database"""
    try:
        if 'current_screen_id' not in session:
            return
        
        screen_id = session['current_screen_id']
        screen = Screen.query.get(screen_id)
        
        if not screen:
            return
        
        # Generate random seat number for demo
        row_letter = chr(65 + random.randint(0, screen.rows - 1))  # A, B, C, etc.
        seat_num = random.randint(1, screen.seats_per_row)
        seat_number = f"{row_letter}{seat_num}"
        
        # Save evidence image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        evidence_dir = "static/evidence"
        os.makedirs(evidence_dir, exist_ok=True)
        evidence_path = f"{evidence_dir}/detection_{timestamp}.jpg"
        
        cv2.imwrite(evidence_path, frame)
        
        # Save to database
        detection = Detection(
            screen_id=screen_id,
            seat_number=seat_number,
            confidence=confidence,
            evidence_path=evidence_path,
            x_coordinate=(x1 + x2) // 2,
            y_coordinate=(y1 + y2) // 2
        )
        
        db.session.add(detection)
        db.session.commit()
        
    except Exception as e:
        print(f"Error saving detection: {e}")

@app.route('/detections/<int:screen_id>')
def get_detections(screen_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    detections = Detection.query.filter_by(screen_id=screen_id).order_by(
        Detection.detection_time.desc()
    ).limit(20).all()
    
    detection_list = []
    for detection in detections:
        detection_list.append({
            'id': detection.id,
            'seat_number': detection.seat_number,
            'time': detection.detection_time.strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': f"{detection.confidence:.2f}",
            'evidence_path': detection.evidence_path,
            'resolved': detection.resolved
        })
    
    return jsonify(detection_list)

@app.route('/seat_map/<int:screen_id>')
def seat_map(screen_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    screen = Screen.query.get_or_404(screen_id)
    recent_detections = Detection.query.filter_by(screen_id=screen_id).filter(
        Detection.detection_time >= datetime.now() - timedelta(hours=1)
    ).all()
    
    try:
        # Create seat map with matplotlib (thread-safe)
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a simple seat layout
        rows = 8
        seats_per_row = 12
        
        # Draw screen
        screen_rect = patches.Rectangle((1, rows + 1), seats_per_row, 0.5, 
                                      linewidth=2, edgecolor='black', facecolor='lightgray')
        ax.add_patch(screen_rect)
        ax.text(seats_per_row/2 + 1, rows + 1.25, 'SCREEN', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Draw seats
        for row in range(rows):
            for seat in range(seats_per_row):
                x = seat + 1.5
                y = rows - row
                
                # Check if this seat has recent detection
                seat_letter = chr(65 + row)
                seat_number = f"{seat_letter}{seat + 1}"
                
                has_detection = any(d.seat_number == seat_number for d in recent_detections)
                
                if has_detection:
                    # Red circle for detected phones
                    circle = patches.Circle((x, y), 0.3, linewidth=2, 
                                          edgecolor='red', facecolor='red', alpha=0.8)
                    ax.add_patch(circle)
                    # Use 'X' instead of emoji
                    ax.text(x, y, 'X', ha='center', va='center', fontsize=12, 
                           color='white', fontweight='bold')
                else:
                    # Blue circle for normal seats
                    circle = patches.Circle((x, y), 0.3, linewidth=1, 
                                          edgecolor='blue', facecolor='lightblue', alpha=0.6)
                    ax.add_patch(circle)
                
                # Seat number
                ax.text(x, y-0.6, f"{seat_letter}{seat+1}", ha='center', va='center', fontsize=8)
        
        # Add random detection for demo if no real detections
        if not recent_detections:
            # Pick random seat for demo
            demo_row = 3
            demo_seat = 7
            x = demo_seat + 1.5
            y = rows - demo_row
            circle = patches.Circle((x, y), 0.3, linewidth=2, 
                                  edgecolor='orange', facecolor='orange', alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, 'X', ha='center', va='center', fontsize=12, 
                   color='white', fontweight='bold')
            ax.text(x, y-0.8, f"D{demo_seat+1} (Demo)", ha='center', va='center', 
                   fontsize=7, color='orange', fontweight='bold')
        
        ax.set_xlim(0, seats_per_row + 2)
        ax.set_ylim(0, rows + 2)
        ax.set_aspect('equal')
        ax.set_title(f'{screen.name} - Phone Detection Map', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            patches.Circle((0, 0), 0.3, facecolor='lightblue', edgecolor='blue', label='Normal Seat'),
            patches.Circle((0, 0), 0.3, facecolor='red', edgecolor='red', label='Phone Detected'),
            patches.Circle((0, 0), 0.3, facecolor='orange', edgecolor='orange', label='Demo Detection')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save plot
        map_dir = "static/seat_maps"
        os.makedirs(map_dir, exist_ok=True)
        map_path = f"{map_dir}/screen_{screen_id}_map.png"
        
        plt.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # Close figure to free memory
        plt.ion()  # Turn interactive mode back on
        
        return send_file(map_path, mimetype='image/png')
        
    except Exception as e:
        print(f"Error generating seat map: {e}")
        # Return a simple error image or placeholder
        return jsonify({'error': 'Failed to generate seat map'}), 500

def init_database():
    """Initialize database with sample data"""
    with app.app_context():
        db.create_all()
        
        # Check if sample data already exists
        if User.query.first():
            return
        
        # Create sample users
        users_data = [
            {
                'username': 'admin',
                'password': 'admin123',
                'theater_name': 'Grand Cinema Palace',
                'email': 'admin@grandcinema.com'
            },
            {
                'username': 'manager',
                'password': 'manager123',
                'theater_name': 'Royal Theater Complex',
                'email': 'manager@royaltheater.com'
            }
        ]
        
        for user_data in users_data:
            user = User(
                username=user_data['username'],
                password_hash=generate_password_hash(user_data['password']),
                theater_name=user_data['theater_name'],
                email=user_data['email']
            )
            db.session.add(user)
        
        db.session.commit()
        
        # Create sample screens
        admin_user = User.query.filter_by(username='admin').first()
        manager_user = User.query.filter_by(username='manager').first()
        
        screens_data = [
            {
                'name': 'Screen 1 - IMAX',
                'total_seats': 120,
                'rows': 8,
                'seats_per_row': 15,
                'user_id': admin_user.id
            },
            {
                'name': 'Screen 2 - Premium',
                'total_seats': 80,
                'rows': 8,
                'seats_per_row': 10,
                'user_id': admin_user.id
            },
            {
                'name': 'Screen A - Standard',
                'total_seats': 100,
                'rows': 10,
                'seats_per_row': 10,
                'user_id': manager_user.id
            }
        ]
        
        for screen_data in screens_data:
            screen = Screen(**screen_data)
            db.session.add(screen)
        
        db.session.commit()
        print("Sample data created successfully!")

@app.route('/test_camera')
def test_camera():
    """Test camera availability"""
    try:
        test_cap = cv2.VideoCapture(0)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            test_cap.release()
            if ret:
                return jsonify({'status': 'Camera working', 'frame_captured': True})
            else:
                return jsonify({'status': 'Camera opened but cannot capture frames', 'frame_captured': False})
        else:
            return jsonify({'status': 'Camera not available', 'frame_captured': False})
    except Exception as e:
        return jsonify({'status': f'Camera error: {str(e)}', 'frame_captured': False})

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Initialize YOLO model
    init_yolo_model()
    
    # Initialize camera manager
    camera_manager = CameraManager()
    
    # Create necessary directories
    os.makedirs('static/evidence', exist_ok=True)
    os.makedirs('static/seat_maps', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)