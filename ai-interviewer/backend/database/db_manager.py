# database/db_manager.py
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, List
import json
import os

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        if connection_string is None:
            connection_string = "mongodb://localhost:27017/"
        
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.db = self.client.interview_db
            
            # Collections
            self.sessions = self.db.sessions
            self.metrics = self.db.metrics
            self.answers = self.db.answers
            
            # Create indexes
            self.sessions.create_index('session_id', unique=True)
            self.metrics.create_index([('session_id', 1), ('timestamp', 1)])
            self.answers.create_index([('session_id', 1), ('question_id', 1)])
            
            # Test connection
            self.client.server_info()
            print("Successfully connected to MongoDB")
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            print("Using file-based storage as fallback")
            self.client = None
            self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Create data directory for file-based storage"""
        os.makedirs('data/sessions', exist_ok=True)
        os.makedirs('data/metrics', exist_ok=True)
        os.makedirs('data/answers', exist_ok=True)

    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Save or update session data"""
        try:
            if self.client:
                self.sessions.update_one(
                    {'session_id': session_id},
                    {'$set': session_data},
                    upsert=True
                )
            else:
                file_path = f'data/sessions/{session_id}.json'
                with open(file_path, 'w') as f:
                    json.dump(session_data, f)
        except Exception as e:
            print(f"Error saving session data: {e}")

    def save_frame_metrics(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Save frame analysis metrics"""
        try:
            metrics['session_id'] = session_id
            metrics['timestamp'] = datetime.now().isoformat()
            
            if self.client:
                self.metrics.insert_one(metrics)
            else:
                file_path = f'data/metrics/{session_id}_metrics.jsonl'
                with open(file_path, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def save_answer(self, session_id: str, answer_data: Dict[str, Any]) -> None:
        """Save interview answer"""
        try:
            answer_data['session_id'] = session_id
            
            if self.client:
                self.answers.insert_one(answer_data)
            else:
                file_path = f'data/answers/{session_id}_answers.jsonl'
                with open(file_path, 'a') as f:
                    f.write(json.dumps(answer_data) + '\n')
        except Exception as e:
            print(f"Error saving answer: {e}")

    def get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session data"""
        try:
            if self.client:
                return self.sessions.find_one({'session_id': session_id})
            else:
                file_path = f'data/sessions/{session_id}.json'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
        except Exception as e:
            print(f"Error retrieving session data: {e}")
        return None

    def get_session_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all metrics for a session"""
        try:
            if self.client:
                return list(self.metrics.find({'session_id': session_id}))
            else:
                file_path = f'data/metrics/{session_id}_metrics.jsonl'
                metrics = []
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        for line in f:
                            metrics.append(json.loads(line))
                return metrics
        except Exception as e:
            print(f"Error retrieving metrics: {e}")
        return []

    def get_session_answers(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all answers for a session"""
        try:
            if self.client:
                return list(self.answers.find({'session_id': session_id}))
            else:
                file_path = f'data/answers/{session_id}_answers.jsonl'
                answers = []
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        for line in f:
                            answers.append(json.loads(line))
                return answers
        except Exception as e:
            print(f"Error retrieving answers: {e}")
        return []

    def update_session(self, session_id: str, update_data: Dict[str, Any]) -> None:
        """Update session data"""
        try:
            if self.client:
                self.sessions.update_one(
                    {'session_id': session_id},
                    {'$set': update_data}
                )
            else:
                current_data = self.get_session_data(session_id) or {}
                current_data.update(update_data)
                self.save_session(session_id, current_data)
        except Exception as e:
            print(f"Error updating session: {e}")

    def cleanup_old_sessions(self, days_old: int = 30) -> None:
        """Clean up old session data"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            if self.client:
                self.sessions.delete_many({'timestamp': {'$lt': cutoff_date}})
                self.metrics.delete_many({'timestamp': {'$lt': cutoff_date}})
                self.answers.delete_many({'timestamp': {'$lt': cutoff_date}})
            else:
                # Clean up file-based storage
                for directory in ['sessions', 'metrics', 'answers']:
                    path = f'data/{directory}'
                    for filename in os.listdir(path):
                        file_path = os.path.join(path, filename)
                        if os.path.getctime(file_path) < cutoff_date:
                            os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")