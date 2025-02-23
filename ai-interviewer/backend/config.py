# backend/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.getenv('DB_NAME', 'interview_db')
    
    # API Keys
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCQSiEPCrsW_RrcDKq1sVCorUigIf2gcx4')
    
    # Interview configuration
    MAX_INTERVIEW_DURATION = int(os.getenv('MAX_INTERVIEW_DURATION', '3600'))  # in seconds
    FRAME_ANALYSIS_FPS = int(os.getenv('FRAME_ANALYSIS_FPS', '30'))
    
    # File storage configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
    
    # Feature flags
    ENABLE_SPEECH_RECOGNITION = os.getenv('ENABLE_SPEECH_RECOGNITION', 'True').lower() == 'true'
    ENABLE_VIDEO_ANALYSIS = os.getenv('ENABLE_VIDEO_ANALYSIS', 'True').lower() == 'true'
    
    # Behavioral analysis thresholds
    SMILE_THRESHOLD = float(os.getenv('SMILE_THRESHOLD', '0.3'))
    FIDGET_THRESHOLD = float(os.getenv('FIDGET_THRESHOLD', '0.1'))
    POSTURE_THRESHOLD = float(os.getenv('POSTURE_THRESHOLD', '0.15'))
    
    @staticmethod
    def init_app(app):
        """Initialize application with config settings"""
        # Create required directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        # Set Flask configs
        app.config['SECRET_KEY'] = Config.SECRET_KEY
        app.config['DEBUG'] = Config.DEBUG
        app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
        app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        
        return app