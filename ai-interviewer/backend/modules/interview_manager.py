# modules/interview_manager.py
from datetime import datetime
import json
from typing import Dict, List, Any
from database.db_manager import DatabaseManager
from .behavior_analysis import BehaviorAnalyzer
from .question_generator import QuestionGenerator

class InterviewManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.question_generator = QuestionGenerator()
        self.active_sessions = {}

    def create_session(self, interview_type: str) -> str:
        """Create a new interview session"""
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        session_data = {
            'session_id': session_id,
            'interview_type': interview_type,
            'start_time': datetime.now().isoformat(),
            'metrics': [],
            'questions': self.question_generator.generate_questions(interview_type),
            'answers': [],
            'status': 'active'
        }
        
        self.active_sessions[session_id] = session_data
        self.db.save_session(session_id, session_data)
        
        return session_id

    def add_frame_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Add frame analysis metrics to the session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        metrics['timestamp'] = datetime.now().isoformat()
        self.active_sessions[session_id]['metrics'].append(metrics)
        self.db.save_frame_metrics(session_id, metrics)

    def evaluate_answer(self, answer: str, question_id: int, session_id: str) -> Dict[str, Any]:
        """Evaluate a user's answer"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        question = session['questions'][question_id]
        
        # Save the answer
        answer_data = {
            'question_id': question_id,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        session['answers'].append(answer_data)
        
        # Evaluate the answer
        evaluation = self._evaluate_answer_quality(answer, question)
        answer_data['evaluation'] = evaluation
        
        self.db.save_answer(session_id, answer_data)
        
        return evaluation

    def _evaluate_answer_quality(self, answer: str, question: str) -> Dict[str, Any]:
        """Internal method to evaluate answer quality"""
        # This is a simplified evaluation. You can make it more sophisticated.
        evaluation = {
            'completeness': self._calculate_completeness(answer),
            'relevance': self._calculate_relevance(answer, question),
            'clarity': self._calculate_clarity(answer),
            'feedback': self._generate_feedback(answer, question)
        }
        
        return evaluation

    def _calculate_completeness(self, answer: str) -> float:
        """Calculate answer completeness score"""
        words = len(answer.split())
        if words < 10:
            return 0.3
        elif words < 30:
            return 0.6
        else:
            return 0.9

    def _calculate_relevance(self, answer: str, question: str) -> float:
        """Calculate answer relevance score"""
        # Simplified relevance calculation
        # You could use more sophisticated NLP techniques here
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        common_words = question_keywords.intersection(answer_keywords)
        
        return len(common_words) / len(question_keywords)

    def _calculate_clarity(self, answer: str) -> float:
        """Calculate answer clarity score"""
        # Simplified clarity calculation
        sentences = answer.split('.')
        avg_words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_words_per_sentence > 25:
            return 0.5  # Too long sentences
        elif avg_words_per_sentence < 5:
            return 0.6  # Too short sentences
        else:
            return 0.9

    def _generate_feedback(self, answer: str, question: str) -> str:
        """Generate feedback for the answer"""
        completeness = self._calculate_completeness(answer)
        relevance = self._calculate_relevance(answer, question)
        clarity = self._calculate_clarity(answer)
        
        feedback = []
        
        if completeness < 0.6:
            feedback.append("Consider providing more detail in your answer.")
        if relevance < 0.5:
            feedback.append("Try to focus more directly on addressing the question.")
        if clarity < 0.7:
            feedback.append("Consider restructuring your answer for better clarity.")
        
        if not feedback:
            feedback.append("Good answer! You've addressed the question well.")
            
        return " ".join(feedback)

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End an interview session and generate summary"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        session['end_time'] = datetime.now().isoformat()
        session['status'] = 'completed'
        
        # Generate summary statistics
        summary = self._generate_session_summary(session)
        session['summary'] = summary
        
        # Save final session data
        self.db.update_session(session_id, session)
        
        # Clean up
        del self.active_sessions[session_id]
        
        return summary

    def _generate_session_summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the session"""
        metrics = session['metrics']
        
        # Calculate behavioral metrics
        smile_intensities = [m.get('smile_intensity', 0) for m in metrics]
        eye_contact_frames = sum(1 for m in metrics if m.get('eye_contact') == 'Maintaining Eye Contact')
        good_posture_frames = sum(1 for m in metrics if m.get('posture') == 'Upright Posture')
        total_frames = len(metrics)
        
        # Calculate answer metrics
        answers = session['answers']
        avg_completeness = sum(a['evaluation']['completeness'] for a in answers) / len(answers) if answers else 0
        avg_relevance = sum(a['evaluation']['relevance'] for a in answers) / len(answers) if answers else 0
        avg_clarity = sum(a['evaluation']['clarity'] for a in answers) / len(answers) if answers else 0
        
        summary = {
            'session_id': session['session_id'],
            'interview_type': session['interview_type'],
            'duration_seconds': (datetime.fromisoformat(session['end_time']) - 
                               datetime.fromisoformat(session['start_time'])).total_seconds(),
            'behavioral_metrics': {
                'average_smile_intensity': sum(smile_intensities) / len(smile_intensities) if smile_intensities else 0,
                'eye_contact_percentage': (eye_contact_frames / total_frames * 100) if total_frames > 0 else 0,
                'good_posture_percentage': (good_posture_frames / total_frames * 100) if total_frames > 0 else 0
            },
            'answer_metrics': {
                'questions_answered': len(answers),
                'average_completeness': avg_completeness,
                'average_relevance': avg_relevance,
                'average_clarity': avg_clarity
            },
            'overall_performance': self._calculate_overall_performance({
                'behavioral': {
                    'smile': sum(smile_intensities) / len(smile_intensities) if smile_intensities else 0,
                    'eye_contact': eye_contact_frames / total_frames if total_frames > 0 else 0,
                    'posture': good_posture_frames / total_frames if total_frames > 0 else 0
                },
                'answers': {
                    'completeness': avg_completeness,
                    'relevance': avg_relevance,
                    'clarity': avg_clarity
                }
            })
        }
        
        return summary

    def _calculate_overall_performance(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        behavioral_score = (
            metrics['behavioral']['smile'] * 0.3 +
            metrics['behavioral']['eye_contact'] * 0.4 +
            metrics['behavioral']['posture'] * 0.3
        )
        
        answer_score = (
            metrics['answers']['completeness'] * 0.4 +
            metrics['answers']['relevance'] * 0.4 +
            metrics['answers']['clarity'] * 0.2
        )
        
        # Weight behavioral and answer scores
        overall_score = behavioral_score * 0.4 + answer_score * 0.6
        
        return round(overall_score * 100, 2)  # Convert to percentage