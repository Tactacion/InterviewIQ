# main.py
import cv2
import numpy as np
from datetime import datetime
import time
import os
from modules.behavior_detector import BehaviorDetector
from modules.behavior_database import BehaviorDatabase

def out():
    # Initialize components
    cap = cv2.VideoCapture(0)
    
    # Debug camera setup
    if not cap.isOpened():
        print("Error: Could not open camera")
        # Try alternative camera indices
        for i in range(1, 5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Successfully opened camera at index {i}")
                break
        if not cap.isOpened():
            print("Could not open any camera")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = BehaviorDetector()
    db = BehaviorDatabase()
    
    # Create session
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Started new session: {session_id}")
    start_time = time.time()
    
    # Initialize storage for all metrics per frame
    all_metrics = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Print status every 30 frames
                print(f"Processing frame {frame_count}")
            
            # Process frame and get all metrics
            metrics, pose_results, face_results, hands_results = detector.analyze_frame(frame)
            
            # Save metrics into our list and into the database
            all_metrics.append(metrics)
            db.save_frame_data(session_id, metrics)
        
            # Display frame
            cv2.imshow('Behavior Analysis', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        session_duration = time.time() - start_time
        
        # Build detailed summary statistics using the collected frame metrics
        if all_metrics:
            smile_values = [m.get('smile_intensity', 0.0) for m in all_metrics]
            total_frames = len(all_metrics)
            average_smile = float(np.mean(smile_values))
            peak_smile = float(np.max(smile_values))
            time_smiling_percentage = float(sum(1 for m in all_metrics if m.get('smile_intensity', 0.0) > 0.3) / total_frames * 100)
            
            posture_analysis = {
                'upright_frames': sum(1 for m in all_metrics if m.get('posture') == 'Upright Posture'),
                'slumped_frames': sum(1 for m in all_metrics if m.get('posture') == 'Slumped Posture'),
                'unknown_frames': sum(1 for m in all_metrics if m.get('posture') == 'Unknown'),
            }
            
            arms_crossed_percentage = float(sum(1 for m in all_metrics if m.get('arms_status') == 'Arms crossed') / total_frames * 100)
            eye_contact_percentage = float(sum(1 for m in all_metrics if m.get('eye_contact') == 'Maintaining Eye Contact') / total_frames * 100)
            total_fidget_instances = sum(1 for m in all_metrics if m.get('fidget_status') == 'Fidgeting Detected')
            fidget_frequency = float(total_fidget_instances / (session_duration / 60))  # per minute
            
            final_summary = {
                
                
                'average_smile_intensity': average_smile,
                
               
                
                'arms_crossed_percentage': arms_crossed_percentage,
                'eye_contact_percentage': eye_contact_percentage,
                'fidgeting_instances': total_fidget_instances,
                
                'total_slumped_time': detector.total_slumped_time,
                
                
            }
        else:
            final_summary = {
                'session_id': session_id,
                'total_frames': 0,
                'session_duration_seconds': session_duration,
                'message': 'No frame metrics were recorded.'
            }
        
        # Save and display the summary
        db.save_session_summary(session_id, final_summary)
        output = []
        print("\nSession Summary:")
        print("=" * 50)
        for key, value in final_summary.items():
            print(f"{key}: {value}")
            output.append(value)
        
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        return output


