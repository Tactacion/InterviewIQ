# behavior_analysis.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from datetime import datetime
import json
import os
from typing import Dict, Tuple, Any, Optional,List

class FacialAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )

        # Define key facial landmark indices
        self.MOUTH_LANDMARKS = {
            'corners': [61, 291],
            'outer_corners': [76, 306],
            'top_outer': [40, 37, 0, 267, 269, 270],
            'top_inner': [415, 419, 396, 174, 172, 165],
            'bottom_outer': [17, 16, 15, 14, 13, 12],
            'bottom_inner': [392, 308, 324, 318, 402, 317],
            'smile_lines': [76, 77, 78, 79, 80, 306, 307, 308, 309, 310],
            'teeth_region': [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        }

        self.smile_history = []
        self.last_update = time.time()
        self.baseline = None
        self.max_mouth_height = None

    def _get_point(self, landmarks, idx):
        point = landmarks[idx]
        return np.array([point.x, point.y, point.z])

    def _initialize_baseline(self, landmarks):
        """Initialize baseline measurements for neutral expression."""
        self.baseline = {
            'mouth_width': self._get_mouth_width(landmarks),
            'mouth_height': self._get_mouth_height(landmarks),
            'corner_elevation': self._get_corner_elevation(landmarks),
            'smile_line_depth': self._get_smile_line_depth(landmarks),
            'teeth_visibility': self._get_teeth_visibility(landmarks)
        }
        self.max_mouth_height = self._get_mouth_height(landmarks)

    def _get_mouth_width(self, landmarks):
        """Calculate enhanced mouth width measurement."""
        main_width = np.linalg.norm(
            self._get_point(landmarks, self.MOUTH_LANDMARKS['corners'][0]) -
            self._get_point(landmarks, self.MOUTH_LANDMARKS['corners'][1])
        )
        outer_width = np.linalg.norm(
            self._get_point(landmarks, self.MOUTH_LANDMARKS['outer_corners'][0]) -
            self._get_point(landmarks, self.MOUTH_LANDMARKS['outer_corners'][1])
        )
        return max(main_width, outer_width)

    def _get_mouth_height(self, landmarks):
        """Calculate mouth opening with inner lip tracking."""
        top_inner = np.mean([self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['top_inner']], axis=0)
        bottom_inner = np.mean([self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['bottom_inner']], axis=0)
        height = np.linalg.norm(top_inner - bottom_inner)
        
        if self.max_mouth_height is None or height > self.max_mouth_height:
            self.max_mouth_height = height
        
        return height

    def _get_corner_elevation(self, landmarks):
        """Enhanced corner elevation detection."""
        left_corner = self._get_point(landmarks, self.MOUTH_LANDMARKS['corners'][0])
        right_corner = self._get_point(landmarks, self.MOUTH_LANDMARKS['corners'][1])
        center_top = np.mean([self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['top_outer'][2:4]], axis=0)
        
        left_elevation = center_top[1] - left_corner[1]
        right_elevation = center_top[1] - right_corner[1]
        
        return (left_elevation + right_elevation) / 2

    def _get_teeth_visibility(self, landmarks):
        """Calculate teeth visibility based on mouth opening and inner lip positions."""
        inner_top_points = [self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['top_inner']]
        inner_bottom_points = [self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['bottom_inner']]
        
        vertical_gaps = [abs(top[1] - bottom[1]) for top, bottom in zip(inner_top_points, inner_bottom_points)]
        avg_gap = np.mean(vertical_gaps)
        
        mouth_region_points = [self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['teeth_region']]
        hull = cv2.convexHull(np.array(mouth_region_points)[:, :2].astype(np.float32))
        area = cv2.contourArea(hull)
        
        return avg_gap * area

    def _get_smile_line_depth(self, landmarks):
        """Enhanced smile line (nasolabial fold) detection."""
        left_points = [self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['smile_lines'][:5]]
        right_points = [self._get_point(landmarks, idx) for idx in self.MOUTH_LANDMARKS['smile_lines'][5:]]
        
        left_depth = np.mean([p[2] for p in left_points])
        right_depth = np.mean([p[2] for p in right_points])
        
        return (left_depth + right_depth) / 2

    def _analyze_smile_intensity(self, landmarks) -> float:
        """Enhanced smile analysis with open mouth and teeth visibility."""
        if self.baseline is None:
            self._initialize_baseline(landmarks)
            return 0.0

        current = {
            'mouth_width': self._get_mouth_width(landmarks),
            'mouth_height': self._get_mouth_height(landmarks),
            'corner_elevation': self._get_corner_elevation(landmarks),
            'smile_line_depth': self._get_smile_line_depth(landmarks),
            'teeth_visibility': self._get_teeth_visibility(landmarks)
        }

        # Calculate weighted feature contributions
        width_factor = (current['mouth_width'] / (self.baseline['mouth_width'] + 1e-5) - 1) * 6
        height_ratio = current['mouth_height'] / (self.baseline['mouth_height'] + 1e-5)
        height_factor = height_ratio * 3 if height_ratio > 1.2 else (1 - height_ratio) * 2
        elevation_factor = (current['corner_elevation'] - self.baseline['corner_elevation']) * 8
        teeth_factor = (current['teeth_visibility'] / (self.baseline['teeth_visibility'] + 1e-5) - 1) * 4
        smile_line_factor = (current['smile_line_depth'] - self.baseline['smile_line_depth']) * 4

        # Combine factors with weights
        smile_intensity = (
            width_factor * 0.35 +
            height_factor * 0.15 +
            elevation_factor * 0.25 +
            teeth_factor * 0.15 +
            smile_line_factor * 0.1
        )

        # Temporal smoothing
        current_time = time.time()
        dt = current_time - self.last_update
        if dt > 0.02:  # Update history at maximum 50Hz
            self.smile_history.append(smile_intensity)
            if len(self.smile_history) > 3:
                self.smile_history.pop(0)
            self.last_update = current_time

        if self.smile_history:
            smile_intensity = (smile_intensity * 0.8 + np.mean(self.smile_history) * 0.2)

        # Normalize to [0, 1] range using tanh
        smile_intensity = np.tanh(smile_intensity * 2.0)
        
        return float(np.clip(smile_intensity, 0, 1))

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze facial features focusing on smile detection."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        metrics = {'smile_intensity': 0.0}
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            metrics['smile_intensity'] = self._analyze_smile_intensity(landmarks)
        
        return metrics, results

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.slumped_start_time = None
        self.total_slumped_time = 0

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze posture in a single frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        metrics = {
            'posture': 'Unknown',
            'slumped_time': self.total_slumped_time
        }
        
        if results.pose_landmarks:
            metrics['posture'] = self._analyze_posture(results.pose_landmarks.landmark)
            
        return metrics, results

    def _analyze_posture(self, landmarks) -> str:
        """Analyzes the posture based on spine alignment."""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # Calculate shoulder midpoint and adjust for natural posture
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        adjusted_shoulder_y = avg_shoulder_y - 0.1  # Adjust threshold for natural posture
        
        # Calculate vertical difference between nose and adjusted shoulder height
        vertical_diff = adjusted_shoulder_y - nose.y
        
        current_time = time.time()
        threshold = 0.15
        
        if vertical_diff < threshold:
            if self.slumped_start_time is None:
                self.slumped_start_time = current_time
            else:
                self.total_slumped_time += (current_time - self.slumped_start_time)
                self.slumped_start_time = current_time
            return "Slumped Posture"
        else:
            self.slumped_start_time = None
            return "Upright Posture"

class HandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.hand_positions = deque(maxlen=10)
        self.last_fidget_time = time.time()
        self.fidget_cooldown = 1.0

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze hand movements and gestures."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        metrics = {'fidget_status': 'No Hands Detected'}
        
        if results.multi_hand_landmarks:
            metrics['fidget_status'] = self._detect_fidgeting(results.multi_hand_landmarks[0].landmark)
            
        return metrics, results

    def _detect_fidgeting(self, hand_landmarks) -> str:
        """Detect rapid hand movements indicating fidgeting."""
        if not hand_landmarks:
            return "No Hands Detected"

        wrist = hand_landmarks[0]
        current_pos = np.array([wrist.x, wrist.y, wrist.z])
        
        self.hand_positions.append(current_pos)
        
        if len(self.hand_positions) >= 2:
            movement = np.linalg.norm(self.hand_positions[-1] - self.hand_positions[-2])
            
            current_time = time.time()
            if movement > 0.1 and current_time - self.last_fidget_time > self.fidget_cooldown:
                self.last_fidget_time = current_time
                return "Fidgeting Detected"
        
        return "No Fidgeting"

class BehaviorAnalyzer:
    def __init__(self):
        self.facial_analyzer = FacialAnalyzer()
        self.posture_analyzer = PostureAnalyzer()
        self.hand_analyzer = HandAnalyzer()

    def analyze_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Analyze all behavioral aspects in a single frame."""
        # Get facial analysis
        facial_metrics, face_results = self.facial_analyzer.analyze_frame(frame)
        
        # Get posture analysis
        posture_metrics, pose_results = self.posture_analyzer.analyze_frame(frame)
        
        # Get hand analysis
        hand_metrics, hand_results = self.hand_analyzer.analyze_frame(frame)
        
        # Combine all metrics
        metrics = {
            **facial_metrics,
            **posture_metrics,
            **hand_metrics
        }
        
        # Return metrics and all analysis results
        return metrics, {
            'face': face_results,
            'pose': pose_results,
            'hands': hand_results
        }

    def cleanup(self):
        """Clean up resources."""
        self.facial_analyzer.face_mesh.close()
        self.posture_analyzer.pose.close()
        self.hand_analyzer.hands.close()

# Continuing BehaviorVisualizer class...

class BehaviorVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        
        # Drawing specifications
        self.pose_landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.pose_connection_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        self.face_landmark_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        self.face_connection_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
        self.hand_landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        self.hand_connection_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)

    def draw_landmarks(self, frame, analysis_results):
        """Draw all landmarks on the frame."""
        # Draw pose landmarks
        if analysis_results['pose'].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                analysis_results['pose'].pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.pose_landmark_spec,
                self.pose_connection_spec
            )

        # Draw face mesh
        if analysis_results['face'].multi_face_landmarks:
            for face_landmarks in analysis_results['face'].multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.face_landmark_spec,
                    self.face_connection_spec
                )

        # Draw hand landmarks
        if analysis_results['hands'].multi_hand_landmarks:
            for hand_landmarks in analysis_results['hands'].multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.hand_landmark_spec,
                    self.hand_connection_spec
                )

    def draw_metrics(self, frame, metrics: Dict[str, any]):
        """Draw behavior metrics on the frame."""
        # Draw smile intensity bar
        smile_intensity = metrics.get('smile_intensity', 0.0)
        bar_length = int(300 * smile_intensity)
        cv2.rectangle(frame, (10, 50), (310, 70), (0, 0, 0), 2)
        cv2.rectangle(frame, (10, 50), (10 + bar_length, 70),
                     (int(255 * (1 - smile_intensity)),
                      int(255 * smile_intensity), 0), -1)

        # Draw metrics text
        feedback = [
            f"Smile Intensity: {metrics['smile_intensity']:.3f}",
            f"Posture: {metrics['posture']}",
            f"Fidgeting: {metrics['fidget_status']}",
            f"Slumped Time: {metrics['slumped_time']:.1f} sec"
        ]

        for i, text in enumerate(feedback):
            cv2.putText(
                frame,
                text,
                (10, 90 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )



class SessionManager:
    def __init__(self, save_path: str = "sessions"):
        """Initialize the session manager."""
        self.save_path = save_path
        self.current_session_id = None
        self.session_start_time = None
        self.frames_processed = 0
        self.session_metrics = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def start_new_session(self) -> str:
        """Start a new recording session."""
        self.current_session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_start_time = datetime.now()
        self.frames_processed = 0
        self.session_metrics = []
        return self.current_session_id

    def save_frame_metrics(self, metrics: Dict[str, Any]):
        """Save metrics from a single frame."""
        if not self.current_session_id:
            raise ValueError("No active session")

        frame_data = {
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.session_metrics.append(frame_data)
        self.frames_processed += 1

    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary statistics."""
        if not self.current_session_id:
            raise ValueError("No active session")

        # Generate detailed summary
        summary = self._generate_detailed_summary()
        
        # Save session data
        self._save_session_data(summary)
        
        # Reset session
        self.current_session_id = None
        self.session_start_time = None
        self.frames_processed = 0
        self.session_metrics = []
        
        return summary

    def _save_session_data(self, summary: Dict[str, Any]):
        """Save session data to files."""
        session_dir = os.path.join(self.save_path, self.current_session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save frame-by-frame metrics
        metrics_file = os.path.join(session_dir, "frame_metrics.jsonl")
        with open(metrics_file, 'w') as f:
            for metrics in self.session_metrics:
                json.dump(metrics, f)
                f.write('\n')
        
        # Save summary
        summary_file = os.path.join(session_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_detailed_summary(self) -> Dict[str, Any]:
        """Generate detailed behavioral analysis summary."""
        if not self.session_metrics:
            return {}
        
        # Calculate time ranges
        timestamps = [m['timestamp'] for m in self.session_metrics]
        start_time = datetime.fromisoformat(timestamps[0])
        end_time = datetime.fromisoformat(timestamps[-1])
        total_duration = (end_time - start_time).total_seconds()
        
        # Split data into thirds for temporal analysis
        first_third = self.session_metrics[:len(self.session_metrics)//3]
        middle_third = self.session_metrics[len(self.session_metrics)//3:2*len(self.session_metrics)//3]
        last_third = self.session_metrics[2*len(self.session_metrics)//3:]

        # Detailed Smile Analysis
        smile_analysis = {
            "overall_metrics": {
                "average_intensity": float(np.mean([m['smile_intensity'] for m in self.session_metrics])),
                "peak_intensity": float(np.max([m['smile_intensity'] for m in self.session_metrics])),
                "time_spent_smiling": float(sum(1 for m in self.session_metrics if m['smile_intensity'] > 0.3) / len(self.session_metrics) * 100),
            },
            "temporal_breakdown": {
                "beginning_phase": {
                    "avg_smile": float(np.mean([m['smile_intensity'] for m in first_third])),
                    "time_smiling": float(sum(1 for m in first_third if m['smile_intensity'] > 0.3) / len(first_third) * 100)
                },
                "middle_phase": {
                    "avg_smile": float(np.mean([m['smile_intensity'] for m in middle_third])),
                    "time_smiling": float(sum(1 for m in middle_third if m['smile_intensity'] > 0.3) / len(middle_third) * 100)
                },
                "end_phase": {
                    "avg_smile": float(np.mean([m['smile_intensity'] for m in last_third])),
                    "time_smiling": float(sum(1 for m in last_third if m['smile_intensity'] > 0.3) / len(last_third) * 100)
                }
            },
            "peak_moments": [
                {
                    "timestamp": m['timestamp'],
                    "intensity": m['smile_intensity']
                }
                for m in self.session_metrics if m['smile_intensity'] > 0.8
            ]
        }

        # Detailed Posture Analysis
        posture_analysis = {
            "overall_metrics": {
                "total_frames": len(self.session_metrics),
                "upright_frames": sum(1 for m in self.session_metrics if m['posture'] == 'Upright Posture'),
                "slumped_frames": sum(1 for m in self.session_metrics if m['posture'] == 'Slumped Posture'),
                "unknown_frames": sum(1 for m in self.session_metrics if m['posture'] == 'Unknown'),
                "total_slumped_time": float(self.session_metrics[-1]['slumped_time'])
            },
            "temporal_breakdown": {
                "beginning_phase": {
                    "upright_percentage": float(sum(1 for m in first_third if m['posture'] == 'Upright Posture') / len(first_third) * 100)
                },
                "middle_phase": {
                    "upright_percentage": float(sum(1 for m in middle_third if m['posture'] == 'Upright Posture') / len(middle_third) * 100)
                },
                "end_phase": {
                    "upright_percentage": float(sum(1 for m in last_third if m['posture'] == 'Upright Posture') / len(last_third) * 100)
                }
            }
        }

        # Arms and Eye Contact Analysis
        body_language_analysis = {
            "arms_position": {
                "crossed_percentage": float(sum(1 for m in self.session_metrics if m['arms_status'] == 'Arms crossed') / len(self.session_metrics) * 100),
                "uncrossed_percentage": float(sum(1 for m in self.session_metrics if m['arms_status'] == 'Arms Uncrossed') / len(self.session_metrics) * 100),
                "temporal_breakdown": {
                    "beginning_phase": {
                        "crossed_percentage": float(sum(1 for m in first_third if m['arms_status'] == 'Arms crossed') / len(first_third) * 100)
                    },
                    "middle_phase": {
                        "crossed_percentage": float(sum(1 for m in middle_third if m['arms_status'] == 'Arms crossed') / len(middle_third) * 100)
                    },
                    "end_phase": {
                        "crossed_percentage": float(sum(1 for m in last_third if m['arms_status'] == 'Arms crossed') / len(last_third) * 100)
                    }
                }
            },
            "eye_contact": {
                "maintained_percentage": float(sum(1 for m in self.session_metrics if m['eye_contact'] == 'Maintaining Eye Contact') / len(self.session_metrics) * 100)
            }
        }

        # Fidgeting Analysis
        fidget_analysis = {
            "overall_metrics": {
                "total_instances": sum(1 for m in self.session_metrics if m['fidget_status'] == 'Fidgeting Detected'),
                "frequency_per_minute": float(sum(1 for m in self.session_metrics if m['fidget_status'] == 'Fidgeting Detected') / (total_duration / 60))
            },
            "temporal_breakdown": {
                "beginning_phase": sum(1 for m in first_third if m['fidget_status'] == 'Fidgeting Detected'),
                "middle_phase": sum(1 for m in middle_third if m['fidget_status'] == 'Fidgeting Detected'),
                "end_phase": sum(1 for m in last_third if m['fidget_status'] == 'Fidgeting Detected')
            }
        }

        # Overall Engagement Analysis
        engagement_analysis = {
            "high_engagement_indicators": {
                "upright_and_smiling": float(sum(1 for m in self.session_metrics 
                    if m['posture'] == 'Upright Posture' 
                    and m['smile_intensity'] > 0.3 
                    and m['arms_status'] == 'Arms Uncrossed') / len(self.session_metrics) * 100),
            },
            "low_engagement_indicators": {
                "slumped_and_not_smiling": float(sum(1 for m in self.session_metrics 
                    if m['posture'] == 'Slumped Posture' 
                    and m['smile_intensity'] < 0.1) / len(self.session_metrics) * 100)
            }
        }

        # Generate behavioral insights
        behavioral_insights = self._generate_behavioral_insights(
            smile_analysis,
            posture_analysis,
            body_language_analysis,
            fidget_analysis,
            engagement_analysis
        )

        return {
            "session_id": self.current_session_id,
            "session_duration_seconds": total_duration,
            "total_frames": len(self.session_metrics),
            "average_fps": len(self.session_metrics) / total_duration,
            "smile_analysis": smile_analysis,
            "posture_analysis": posture_analysis,
            "body_language_analysis": body_language_analysis,
            "fidget_analysis": fidget_analysis,
            "engagement_analysis": engagement_analysis,
            "behavioral_insights": behavioral_insights,
            "session_timestamps": {
                "start": timestamps[0],
                "end": timestamps[-1]
            }
        }

    def _generate_behavioral_insights(self, smile_analysis, posture_analysis, 
                                   body_language_analysis, fidget_analysis, 
                                   engagement_analysis) -> List[str]:
        """Generate human-readable insights from the metrics."""
        insights = []
        
        # Smile insights
        avg_smile = smile_analysis["overall_metrics"]["average_intensity"]
        time_smiling = smile_analysis["overall_metrics"]["time_spent_smiling"]
        if avg_smile > 0.5:
            insights.append("Candidate showed genuine, consistent positive expression throughout the interview")
        elif time_smiling > 30:
            insights.append("Candidate maintained positive expression for significant portions of the interview")
        elif avg_smile < 0.2:
            insights.append("Candidate maintained mostly neutral expression throughout the interview")

        # Posture insights
        upright_percentage = (posture_analysis["overall_metrics"]["upright_frames"] / 
                            posture_analysis["overall_metrics"]["total_frames"] * 100)
        if upright_percentage > 80:
            insights.append("Excellent posture maintained throughout the session")
        elif upright_percentage > 60:
            insights.append("Generally good posture with some periods of slouching")
        else:
            insights.append("Significant room for improvement in maintaining upright posture")

        # Body language insights
        arms_uncrossed = body_language_analysis["arms_position"]["uncrossed_percentage"]
        eye_contact = body_language_analysis["eye_contact"]["maintained_percentage"]
        if arms_uncrossed > 80 and eye_contact > 90:
            insights.append("Very open and engaged body language with excellent eye contact")
        elif eye_contact > 90:
            insights.append("Maintained strong eye contact throughout the session")
        
        # Fidgeting insights
        fidget_freq = fidget_analysis["overall_metrics"]["frequency_per_minute"]
        if fidget_freq < 1:
            insights.append("Displayed composed and calm demeanor with minimal fidgeting")
        elif fidget_freq > 3:
            insights.append("Shows signs of nervousness through frequent fidgeting")

        # Engagement insights
        high_engagement = engagement_analysis["high_engagement_indicators"]["upright_and_smiling"]
        if high_engagement > 70:
            insights.append("Demonstrated consistently high engagement throughout the session")
        elif high_engagement > 40:
            insights.append("Showed moderate levels of engagement with room for improvement")

        return insights

    def get_session_metrics(self) -> List[Dict[str, Any]]:
        """Return the current session's metrics."""
        return self.session_metrics

    def get_session_duration(self) -> float:
        """Return the current session duration in seconds."""
        if not self.session_start_time:
            return 0.0
        return (datetime.now() - self.session_start_time).total_seconds()
def main():
    """Main function to run the behavior analysis system."""
    # Initialize components
    cap = cv2.VideoCapture(0)
    analyzer = BehaviorAnalyzer()
    visualizer = BehaviorVisualizer()
    session_manager = SessionManager()
    
    # Start session
    session_id = session_manager.start_new_session()
    print(f"Started new session: {session_id}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            metrics, analysis_results = analyzer.analyze_frame(frame)
            
            # Print metrics
            print(metrics)
            
            # Save metrics
            session_manager.save_frame_metrics(metrics)
            
            # Draw visualizations
            visualizer.draw_landmarks(frame, analysis_results)
            visualizer.draw_metrics(frame, metrics)
            
            # Display frame
            cv2.imshow('Behavior Analysis', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during processing: {e}")
        
    finally:
        # End session and get summary
        summary = session_manager.end_session()
        print("\nSession Summary:")
        print(json.dumps(summary, indent=2))
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        analyzer.cleanup()

if __name__ == "__main__":
    main()