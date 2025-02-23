import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from typing import Dict, Tuple

class BehaviorAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )

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

        width_factor = (current['mouth_width'] / (self.baseline['mouth_width'] + 1e-5) - 1) * 6
        height_ratio = current['mouth_height'] / (self.baseline['mouth_height'] + 1e-5)
        height_factor = height_ratio * 3 if height_ratio > 1.2 else (1 - height_ratio) * 2
        elevation_factor = (current['corner_elevation'] - self.baseline['corner_elevation']) * 8
        teeth_factor = (current['teeth_visibility'] / (self.baseline['teeth_visibility'] + 1e-5) - 1) * 4
        smile_line_factor = (current['smile_line_depth'] - self.baseline['smile_line_depth']) * 4

        smile_intensity = (
            width_factor * 0.35 +
            height_factor * 0.15 +
            elevation_factor * 0.25 +
            teeth_factor * 0.15 +
            smile_line_factor * 0.1
        )

        current_time = time.time()
        dt = current_time - self.last_update
        if dt > 0.02:
            self.smile_history.append(smile_intensity)
            if len(self.smile_history) > 3:
                self.smile_history.pop(0)
            self.last_update = current_time

        if self.smile_history:
            smile_intensity = (smile_intensity * 0.8 + np.mean(self.smile_history) * 0.2)

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

class BehaviorDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.facial_analyzer = BehaviorAnalyzer()
        
        self.hand_positions = deque(maxlen=10)
        self.last_fidget_time = time.time()
        self.fidget_cooldown = 1.0
        self.slumped_start_time = None
        self.total_slumped_time = 0

    def analyze_frame(self, frame: np.ndarray) -> Tuple[Dict, any, any, any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with all detectors
        pose_results = self.pose.process(rgb_frame)
        smile_metrics, face_results = self.facial_analyzer.analyze_frame(frame)
        hands_results = self.hands.process(rgb_frame)
        
        metrics = {
            'smile_intensity': smile_metrics['smile_intensity'],
            'posture': 'Unknown',
            'eye_contact': 'No Face Detected',
            'arms_status': 'Unknown',
            'fidget_status': 'No Hands Detected',
            'slumped_time': self.total_slumped_time
        }
        
        if pose_results.pose_landmarks:
            metrics['posture'] = self.analyze_posture(pose_results.pose_landmarks.landmark)
            metrics['arms_status'] = self.detect_crossed_arms(pose_results.pose_landmarks.landmark)
            
        if face_results.multi_face_landmarks:
            metrics['eye_contact'] = self.detect_eye_contact(
                face_results.multi_face_landmarks[0].landmark, 
                frame.shape[1]
            )
            
        if hands_results.multi_hand_landmarks:
            metrics['fidget_status'] = self.detect_hand_fidgeting(
                hands_results.multi_hand_landmarks[0].landmark
            )
        
        return metrics, pose_results, face_results, hands_results

    def analyze_posture(self, landmarks):
        """Analyzes the posture based on the vertical distance between the nose and shoulders."""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        adjusted_shoulder_y = avg_shoulder_y - 0.1
        
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

    def detect_eye_contact(self, face_landmarks, frame_width):
        """Enhanced eye contact detection with head pose estimation"""
        if not face_landmarks:
            return "No Face Detected"

        left_eye = face_landmarks[159]
        right_eye = face_landmarks[145]
        nose_tip = face_landmarks[4]
        
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_distance = abs(left_eye.x - right_eye.x)
        
        if abs(eye_center_x - 0.5) > 0.2 or eye_distance < 0.05:
            return "Maintaining Eye Contact"
        return "idhar dekh laude"

    def detect_crossed_arms(self, landmarks):
        """Enhanced crossed arms detection using multiple joint positions"""
        if not landmarks:
            return "No Pose Detected"

        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        wrists_crossed = (
            left_wrist.x > right_wrist.x and
            left_wrist.y > left_shoulder.y and
            right_wrist.y > right_shoulder.y
        )

        elbows_raised = (
            left_elbow.y < left_shoulder.y and
            right_elbow.y < right_shoulder.y
        )

        if wrists_crossed and not elbows_raised:
            return "Arms Uncrossed"
        return "Arms crossed"

    def detect_hand_fidgeting(self, hand_landmarks):
        """Detect rapid hand movements indicating fidgeting"""
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

    def cleanup(self):
        self.pose.close()
        self.hands.close()
        self.facial_analyzer.face_mesh.close()