# modules/question_generator.py
import google.generativeai as genai
from typing import List
import os

class QuestionGenerator:
    def __init__(self):
        # Initialize the Gemini API with your key
        self.API_KEY = os.getenv('GEMINI_API_KEY', "AIzaSyCQSiEPCrsW_RrcDKq1sVCorUigIf2gcx4")
        genai.configure(api_key=self.API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")

        # Predefined questions for fallback
        self.default_questions = {
            'technical': [
                "Explain the concept of RESTful APIs and their key principles.",
                "What is the difference between HTTP GET and POST methods?",
                "Explain the concept of database indexing and its importance.",
                "What are the principles of Object-Oriented Programming?",
                "Describe the differences between TCP and UDP protocols."
            ],
            'behavioral': [
                "Tell me about a challenging project you worked on.",
                "How do you handle conflicts in a team setting?",
                "Describe a situation where you had to learn something quickly.",
                "How do you prioritize tasks when you have multiple deadlines?",
                "Tell me about a time you failed and what you learned from it."
            ],
            'system_design': [
                "Design a URL shortening service like bit.ly.",
                "How would you design Twitter's backend?",
                "Design a real-time chat application.",
                "How would you design a file-sharing service?",
                "Design an online booking system for a movie theater."
            ]
        }

    def generate_questions(self, interview_type: str) -> List[str]:
        """Generate interview questions based on the type"""
        try:
            # Format the prompt based on interview type
            prompt = self._create_prompt(interview_type)
            
            # Generate questions using Gemini
            response = self.model.generate_content(prompt)
            
            # Process and clean the response
            questions = self._process_response(response.text)
            
            # Validate and ensure we have enough questions
            if len(questions) >= 5:
                return questions[:5]  # Return only first 5 questions
            else:
                # Fall back to default questions if generation fails
                return self._get_default_questions(interview_type)
                
        except Exception as e:
            print(f"Error generating questions: {e}")
            return self._get_default_questions(interview_type)

    def _create_prompt(self, interview_type: str) -> str:
        """Create an appropriate prompt based on interview type"""
        base_prompt = "Generate 5 professional interview questions"
        
        type_specific_prompts = {
            'technical': f"{base_prompt} for a software engineering position. Focus on fundamental concepts and problem-solving abilities. Questions should be technical but not require coding.",
            'behavioral': f"{base_prompt} that assess soft skills, leadership, and past experiences. Questions should help evaluate the candidate's interpersonal skills and problem-solving approach.",
            'system_design': f"{base_prompt} about system design and architecture. Questions should focus on scalability, reliability, and real-world applications."
        }
        
        return type_specific_prompts.get(
            interview_type.lower(), 
            f"{base_prompt} for a {interview_type} interview."
        )

    def _process_response(self, response: str) -> List[str]:
        """Process and clean the generated response"""
        # Split response into lines
        lines = response.strip().split('\n')
        
        # Clean and filter questions
        questions = []
        for line in lines:
            # Remove common prefixes
            line = line.strip()
            line = line.lstrip('0123456789.- )')
            line = line.strip()
            
            # Add if line is long enough to be a question
            if len(line) > 10:
                questions.append(line)
        
        return questions

    def _get_default_questions(self, interview_type: str) -> List[str]:
        """Get default questions for the specified interview type"""
        interview_type = interview_type.lower()
        
        # Map variations to our main categories
        type_mapping = {
            'tech': 'technical',
            'coding': 'technical',
            'programming': 'technical',
            'behavioral': 'behavioral',
            'hr': 'behavioral',
            'system': 'system_design',
            'design': 'system_design',
            'architecture': 'system_design'
        }
        
        # Get the standardized type
        standard_type = type_mapping.get(interview_type, 'behavioral')
        
        # Return the default questions for this type
        return self.default_questions.get(standard_type, self.default_questions['behavioral'])

    def get_ideal_answers(self, question: str) -> str:
        """Generate ideal answer guidelines for a question"""
        try:
            prompt = f"Provide a comprehensive but concise ideal answer for the interview question: {question}"
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating ideal answer: {e}")
            return "Unable to generate ideal answer. Please provide your best response based on your knowledge and experience."