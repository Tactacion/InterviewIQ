# behavior_database.py
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any
import json

class BehaviorDatabase:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """Initialize connection to MongoDB."""
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.server_info()
            self.db = self.client.interview_db
            self.behavior_data = self.db.behavior_data
            self.session_summaries = self.db.session_summaries
            
            # Create indexes
            self.behavior_data.create_index([("session_id", 1), ("timestamp", 1)])
            self.session_summaries.create_index("session_id", unique=True)
            print("Successfully connected to MongoDB")
        except Exception as e:
            print(f"Failed to connect to MongoDB. Error: {e}")
            print("Falling back to file-based storage")
            self.client = None
    
    def save_frame_data(self, session_id: str, frame_data: Dict[str, Any]) -> dict:
        """Save behavioral data from a single frame."""
        record = {
            
            "timestamp": datetime.now().isoformat(),
            "smile_intensity": frame_data.get("smile_intensity", 0.0),
            
            "eye_contact": frame_data.get("eye_contact", "Unknown"),
            "arms_status": frame_data.get("arms_status", "Unknown"),
            "fidget_status": frame_data.get("fidget_status", "Unknown"),
            "slumped_time": frame_data.get("slumped_time", 0.0)
        }
        
        if self.client:
            try:
                self.behavior_data.insert_one(record)
                return record
            except Exception as e:
                print(f"Failed to save frame data to MongoDB: {e}")
                self._save_to_file(record, f"frame_data_{session_id}.json")
        else:
            self._save_to_file(record, f"frame_data_{session_id}.json")

    def save_session_summary(self, session_id: str, summary_data: Dict[str, Any]) -> None:
        """Save summary data for an entire session."""
        summary_record = {
            "session_id": session_id,
            "end_time": datetime.now().isoformat(),
            "summary_stats": summary_data
        }
        
        if self.client:
            try:
                self.session_summaries.update_one(
                    {"session_id": session_id},
                    {"$set": summary_record},
                    upsert=True
                )
            except Exception as e:
                print(f"Failed to save session summary to MongoDB: {e}")
                self._save_to_file(summary_record, f"session_summary_{session_id}.json")
        else:
            self._save_to_file(summary_record, f"session_summary_{session_id}.json")

    def _save_to_file(self, data: Dict, filename: str) -> None:
        """Fallback method to save data to a JSON file."""
        try:
            with open(filename, 'a') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            print(f"Failed to save to file {filename}: {e}")
