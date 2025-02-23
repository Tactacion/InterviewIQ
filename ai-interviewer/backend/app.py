from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import os
import base64
import cv2
import numpy as np
from datetime import datetime
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from modules.behavior_analysis import BehaviorAnalyzer, BehaviorVisualizer, SessionManager
# NOT from behavior_detector import BehaviorDetector
from modules.behavior_database import BehaviorDatabase
from modules.evaluation_module import advanced_analyze_answer_with_explanation,generate_improvement_tips,generate_interview_feedback,calculate_behavioral_metrics

import mediapipe as mp
from modules.speech_processor import text as speech_to_text, speech as text_to_speech


app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
CORS(app)

# Initialize components using your original classes
analyzer = BehaviorAnalyzer()  # Your behavior analyzer
behavior_db = BehaviorDatabase() # Your behavior database

engine = pyttsx3.init()

# Configure Gemini
genai.configure(api_key="AIzaSyCQSiEPCrsW_RrcDKq1sVCorUigIf2gcx4")
model = genai.GenerativeModel("gemini-pro")

# Store active sessions
sessions = {}

@app.route('/')
def index():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.get_json()
    interview_type = data.get('type', 'general')
    
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate questions using Gemini
    prompt = f"Generate 5 professional interview questions for a {interview_type} interview. Questions should be challenging but fair."
    response = model.generate_content(prompt)
    questions = response.text.split('\n')
    questions = [q.strip() for q in questions if q.strip()]
    
    # Initialize session
    sessions[session_id] = {
        'type': interview_type,
        'questions': questions,
        'metrics': [],
        'start_time': datetime.now().isoformat()
    }
   
    # Speak first question
    text_to_speech(questions[0])
    time.sleep(1)
    
    return jsonify({
        'session_id': session_id,
        'questions': questions
    })
@app.route('/get_message', methods=['GET'])
def get_message():
    # Replace this with your actual speech-to-text processing logic
    sptt=speech_to_text()
    speech_result = sptt
    return jsonify({"message": speech_result})
    
    
    

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.get_json()
    session_id = data.get('session_id')
    frame_data = data.get('frame')
    
    if not frame_data or not session_id:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        # Decode base64 image
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use your original analyzer
        metrics, analysis_results = analyzer.analyze_frame(frame)
        
        # Save metrics
        if session_id in sessions:
            sessions[session_id]['metrics'].append(metrics)
        
        # Save to your database
        cv_out = behavior_db.save_frame_data(session_id, metrics)
        
        return jsonify(metrics)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate_answer', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    answer = data.get('answer')
    question_id = data.get('question_id')
    session_id = data.get('session_id')
    
    if not all([answer, session_id]):
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        # Get the question
        question = sessions[session_id]['questions'][question_id]
        
        # Generate LLM feedback
        prompt = f"""Evaluate this interview answer for the question: "{question}"
        Answer: "{answer}"
        Provide constructive feedback and suggestions for improvement."""
        
        response = model.generate_content(prompt)
        feedback = response.text
        
        # Save to your chat database
       
        
        # Speak feedback
        text_to_speech(feedback)
        
        return jsonify({
            'feedback': feedback
        })
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech_to_text', methods=['POST'])
def handle_speech_to_text():
    try:
        text = speech_to_text()
        return jsonify({'text': text})
    except Exception as e:
        print(f"Error in speech to text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/print_summary', methods=['POST'])
def print_summary():
    data = request.json
    session_id = data['session_id']
    chat_history = data['chat_history']
    
    # Get answers and questions only
    answers = [msg['text'] for msg in chat_history if msg['isUser']]
    
    # Use your LLM to generate a summary
    summary = generate_improvement_tips(
        chat_history=chat_history,
        session_id=session_id
    )
    
    # Return the text directly instead of JSON
    return summary
from flask import jsonify
import time

@app.route('/api/end_session', methods=['POST'])
def end_session():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        # Get the chat messages
        chat_messages = data.get('chat_history', [])
        
        # Filter user messages
        user_messages = [msg['text'] for msg in chat_messages if msg.get('isUser', False)]
        
        # Calculate behavioral metrics from the last few frames
        behavioral_metrics = {
            'smile_intensity': 0.75,  # Replace with actual metrics from your OpenCV analysis
            'eye_contact_percentage': 85.0,
            'posture_percentage': 90.0,
            'fidgeting_instances': 2
        }

        # Use Gemini to analyze answers
        model = genai.GenerativeModel("gemini-pro")
        
        analysis_prompt = f"""
        Analyze this technical interview performance:
        
        Interview Duration: {data.get('duration', '00:00:00')}
        Candidate's Responses: {' '.join(user_messages)}
        
        Behavioral Metrics:
        - Smile Engagement: {behavioral_metrics['smile_intensity']*100}%
        - Eye Contact: {behavioral_metrics['eye_contact_percentage']}%
        - Posture: {behavioral_metrics['posture_percentage']}%
        - Fidgeting Instances: {behavioral_metrics['fidgeting_instances']}
        
        Provide a comprehensive analysis covering:
        1. Overall Communication Skills
        2. Body Language and Engagement
        3. Response Quality
        4. Specific Recommendations for Improvement
        
        Format the response in clear sections with bullet points.
        """
        
        analysis_response = model.generate_content(analysis_prompt)
        
        feedback = {
            'session_id': session_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': behavioral_metrics,
            'analysis': analysis_response.text,
            'overall_score': (
                behavioral_metrics['smile_intensity'] * 25 +
                behavioral_metrics['eye_contact_percentage'] * 0.35 +
                behavioral_metrics['posture_percentage'] * 0.4
            )
        }
        
        return jsonify(feedback)

    except Exception as e:
        print(f"Error in end_session: {str(e)}")
        return jsonify({
            'error': 'Failed to generate interview feedback',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5004)