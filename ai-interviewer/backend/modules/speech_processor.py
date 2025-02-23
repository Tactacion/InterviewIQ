# modules/speech_processor.py
import speech_recognition as sr
import pyttsx3
from typing import Optional
import threading
import queue
import tempfile
import os
import wave

# speech_module.py
import speech_recognition as sr
import pyttsx3

def text():
    """
    Converts speech to text using the default microphone and Google Speech API.
    
    Returns:
        str: The transcribed text, or an empty string if the speech is not recognized.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak something...")
        audio_data = recognizer.listen(source)
        try:
            user_text = recognizer.recognize_google(audio_data)
            return user_text
        except sr.UnknownValueError:
            print("Could not understand audio.")
        except sr.RequestError as e:
            print(f"API error: {e}")
    return ""

def speech(text_to_speak):
    """
    Converts text to speech and plays it through the system's speakers.
    
    Args:
        text_to_speak (str): The text that needs to be spoken.
    """
    engine = pyttsx3.init()
    engine.say(text_to_speak)
    engine.runAndWait()
