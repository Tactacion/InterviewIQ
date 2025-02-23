from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
import textstat
from rouge_score import rouge_scorer

# -------------------------------
# TF-IDF Cosine Similarity Comparison
# -------------------------------
def compare_tfidf(user_answer: str, ideal_answer: str) -> (float, float):
    """
    Compute the TF-IDF cosine similarity between the user's answer and the ideal answer.
    
    Returns:
        similarity: Cosine similarity score (0 to 1)
        deviation: 1 - similarity
    """
    documents = [user_answer, ideal_answer]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    deviation = 1 - similarity
    return similarity, deviation

# -------------------------------
# Additional Comparison Features
# -------------------------------
def jaccard_similarity(str1: str, str2: str) -> float:
    """
    Compute the Jaccard similarity between two texts (0 to 1), based on word overlap.
    """
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0.0

def fuzzy_similarity(str1: str, str2: str) -> float:
    """
    Compute fuzzy string matching similarity (0 to 1) based on character alignment.
    """
    return fuzz.ratio(str1, str2) / 100.0

# Load a Sentence Transformer model (this may take a few seconds)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(str1: str, str2: str) -> float:
    """
    Compute semantic similarity (0 to 1) using sentence embeddings.
    """
    embedding1 = semantic_model.encode(str1, convert_to_tensor=True)
    embedding2 = semantic_model.encode(str2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    return cosine_scores.item()

def readability_scores(str1: str, str2: str) -> (float, float, float):
    """
    Compute Flesch reading ease scores for both texts and return both scores and their absolute difference.
    Note: Flesch reading ease scores are typically interpreted on a 0–100 scale.
    """
    score1 = textstat.flesch_reading_ease(str1)
    score2 = textstat.flesch_reading_ease(str2)
    diff = abs(score1 - score2)
    return score1, score2, diff

def rouge_scores(user_answer: str, ideal_answer: str) -> dict:
    """
    Compute ROUGE-1 and ROUGE-L F1 scores between the two texts (0 to 1).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(user_answer, ideal_answer)
    return scores

# -------------------------------
# Generate Improvement Tips
# -------------------------------
def generate_improvement_tips(overall_score: float) -> str:
    """
    Generate tips for improving interview answers based on the overall score.
    """
    if overall_score < 0.6:
        return (
            "Your response has room for improvement. Consider the following tips:\n"
            "- Elaborate more on your answers with specific examples.\n"
            "- Organize your thoughts clearly and maintain a logical structure.\n"
            "- Practice articulating your strengths and experiences succinctly.\n"
            "- Review ideal answers and note areas where you can add depth."
        )
    elif overall_score < 0.8:
        return (
            "Your answer is solid but could be refined further. Here are some suggestions:\n"
            "- Add more detailed examples or case studies to support your points.\n"
            "- Work on clarifying and structuring your answers for greater impact.\n"
            "- Practice with mock interviews to improve your delivery and timing."
        )
    else:
        return (
            "Excellent work! Your answer closely matches the ideal response. To further enhance your skills:\n"
            "- Keep practicing to maintain clarity and consistency.\n"
            "- Work on refining minor details and exploring additional advanced topics.\n"
            "- Consider engaging in mock interviews to simulate real-time feedback."
        )
import google.generativeai as genai

# Set up the Gemini API key
API_KEY = "AIzaSyCQSiEPCrsW_RrcDKq1sVCorUigIf2gcx4"  # Replace with your Gemini API key
genai.configure(api_key=API_KEY)

def generate_interview_questions(interview_type: str):
    """
    Generates 5 interview questions based on the specified interview type.
    """
    prompt = f"Generate 5 high-quality interview questions for a {interview_type}. Just questions nothing else. And do not indicate it with numbers"

    # Call the Gemini LLM API
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    # Extract and display the generated questions
    questions = response.text.split("\n")  # Splitting if Gemini returns as a list

    return questions



# -------------------------------
# Combined Analysis Function with Overall Score and Tips
# -------------------------------
def calculate_behavioral_metrics(behavioral_data):
    """
    Calculate metrics from behavioral data collected during the interview.
    
    Args:
        behavioral_data (dict): Dictionary containing behavioral metrics
        
    Returns:
        dict: Processed behavioral metrics
    """
    # Count total frames
    total_frames = len(behavioral_data) if isinstance(behavioral_data, list) else 1
    
    # Calculate smile intensity average
    smile_intensities = [m.get('smile_intensity', 0) for m in behavioral_data] if isinstance(behavioral_data, list) else [behavioral_data.get('smile_intensity', 0)]
    avg_smile = sum(smile_intensities) / len(smile_intensities)
    
    # Calculate eye contact percentage
    eye_contact_frames = sum(1 for m in behavioral_data if m.get('eye_contact') == 'Maintaining Eye Contact') if isinstance(behavioral_data, list) else (1 if behavioral_data.get('eye_contact') == 'Maintaining Eye Contact' else 0)
    eye_contact_percentage = (eye_contact_frames / total_frames * 100) if total_frames > 0 else 0
    
    # Calculate posture percentage
    good_posture_frames = sum(1 for m in behavioral_data if m.get('posture') == 'Upright Posture') if isinstance(behavioral_data, list) else (1 if behavioral_data.get('posture') == 'Upright Posture' else 0)
    posture_percentage = (good_posture_frames / total_frames * 100) if total_frames > 0 else 0
    
    # Count fidgeting instances
    fidget_count = sum(1 for m in behavioral_data if m.get('fidget_status') == 'Fidgeting Detected') if isinstance(behavioral_data, list) else (1 if behavioral_data.get('fidget_status') == 'Fidgeting Detected' else 0)
    
    return {
        'average_smile_intensity': avg_smile,
        'eye_contact_percentage': eye_contact_percentage,
        'posture_percentage': posture_percentage,
        'fidgeting_instances': fidget_count
    }

def generate_interview_feedback(interview_type: str, chat_history: list, questions: list, questions_completed: int, duration: str, behavioral_metrics: dict = None) -> str:
    """
    Generate comprehensive interview feedback using behavioral metrics and chat analysis.
    
    Args:
        interview_type (str): Type of interview conducted
        chat_history (list): List of chat messages with text and isUser flag
        questions (list): List of questions asked
        questions_completed (int): Number of questions completed
        duration (str): Interview duration
        behavioral_metrics (dict): Dictionary of behavioral metrics
        
    Returns:
        str: Formatted interview feedback
    """
    # Get user answers only
    user_answers = [msg['text'] for msg in chat_history if msg.get('isUser', False)]
    
    # Calculate answer quality metrics
    answer_scores = []
    for q, a in zip(questions[:questions_completed], user_answers):
        # Get ideal answer using Gemini
        model = genai.GenerativeModel("gemini-pro")
        ideal_response = model.generate_content(f"Provide an ideal answer for the interview question: {q}")
        ideal_answer = ideal_response.text
        
        # Analyze answer quality
        metrics = advanced_analyze_answer_with_explanation(a, ideal_answer)
        answer_scores.append(metrics)
    
    # Generate overall feedback using Gemini
    feedback_prompt = f"""
    Generate comprehensive interview feedback for a {interview_type} interview with the following details:
    
    Duration: {duration}
    Questions Completed: {questions_completed} out of {len(questions)}
    
    Behavioral Metrics:
    - Average Smile Intensity: {behavioral_metrics['average_smile_intensity']*100:.1f}%
    - Eye Contact Maintained: {behavioral_metrics['eye_contact_percentage']:.1f}%
    - Good Posture Maintained: {behavioral_metrics['posture_percentage']:.1f}%
    - Fidgeting Instances: {behavioral_metrics['fidgeting_instances']}
    
    Provide specific feedback addressing:
    1. Overall performance
    2. Behavioral strengths and areas for improvement
    3. Answer quality and communication style
    4. Specific recommendations for improvement
    
    Format the feedback in a clear, structured way with sections and bullet points.
    """
    
    model = genai.GenerativeModel("gemini-pro")
    feedback_response = model.generate_content(feedback_prompt)
    llm_feedback = feedback_response.text
    
    # Calculate overall performance score
    behavioral_score = (
        behavioral_metrics['average_smile_intensity'] * 0.3 +
        (behavioral_metrics['eye_contact_percentage'] / 100) * 0.4 +
        (behavioral_metrics['posture_percentage'] / 100) * 0.3
    ) * 100

    answer_quality_score = sum(score['overall_score'] for score in answer_scores) / len(answer_scores) if answer_scores else 0
    overall_score = behavioral_score * 0.4 + answer_quality_score * 0.6

    # Format the complete feedback
    final_feedback = f"""
=== Interview Performance Summary ===
Interview Type: {interview_type}
Duration: {duration}
Questions Completed: {questions_completed}/{len(questions)}
Overall Performance Score: {overall_score:.1f}%

=== Behavioral Analysis ===
• Smile Engagement: {behavioral_metrics['average_smile_intensity']*100:.1f}%
• Eye Contact: {behavioral_metrics['eye_contact_percentage']:.1f}%
• Posture: {behavioral_metrics['posture_percentage']:.1f}%
• Fidgeting Instances: {behavioral_metrics['fidgeting_instances']}

=== Answer Analysis ===
Average Answer Quality: {answer_quality_score:.1f}%

=== Detailed Feedback ===
{llm_feedback}

=== Key Recommendations ===
1. {'Maintain current performance' if overall_score > 80 else 'Focus on improvement'}
2. {'Continue strong eye contact' if behavioral_metrics['eye_contact_percentage'] > 70 else 'Work on maintaining better eye contact'}
3. {'Keep up the positive expression' if behavioral_metrics['average_smile_intensity'] > 0.6 else 'Try to appear more engaged and positive'}
4. {'Excellent posture maintained' if behavioral_metrics['posture_percentage'] > 80 else 'Focus on maintaining better posture'}
"""

    return final_feedback
def advanced_analyze_answer_with_explanation(user_answer: str, ideal_answer: str):
    # Compute individual metrics
    tfidf_similarity, tfidf_deviation = compare_tfidf(user_answer, ideal_answer)
    jaccard = jaccard_similarity(user_answer, ideal_answer)
    fuzzy = fuzzy_similarity(user_answer, ideal_answer)
    sem_sim = semantic_similarity(user_answer, ideal_answer)
    readability_user, readability_ideal, readability_diff = readability_scores(user_answer, ideal_answer)
    rouge = rouge_scores(user_answer, ideal_answer)
    
    # Calculate overall score as an average of selected similarity metrics
    overall_score = (
        tfidf_similarity + jaccard + fuzzy + sem_sim + rouge['rouge1'].fmeasure + rouge['rougeL'].fmeasure
    ) / 6
    
    # Print detailed analysis with absolute values and explanations
    print("=== Detailed Comparison Metrics with Explanations ===\n")
    
    print(f"TF-IDF Cosine Similarity: {tfidf_similarity:.2f} ({tfidf_similarity*100:.0f}/100)")
    print("  *Measures overall text similarity based on term frequency and importance. Higher scores indicate greater similarity.\n")
    
    print(f"Jaccard Similarity: {jaccard:.2f} ({jaccard*100:.0f}/100)")
    print("  *Measures the overlap of unique words between texts (intersection over union). Higher values indicate more common vocabulary.\n")
    
    print(f"Fuzzy Matching Score: {fuzzy:.2f} ({fuzzy*100:.0f}/100)")
    print("  *Assesses similarity based on character alignment and edit distance. A score closer to 1 means near-identical strings.\n")
    
    print(f"Semantic Similarity: {sem_sim:.2f} ({sem_sim*100:.0f}/100)")
    print("  *Uses sentence embeddings to capture the underlying meaning of the texts. Higher values indicate semantically similar answers.\n")
    
    print(f"User Answer Readability (Flesch Reading Ease): {readability_user:.2f} (scale 0-100)")
    print(f"Ideal Answer Readability (Flesch Reading Ease): {readability_ideal:.2f} (scale 0-100)")
    print(f"Readability Difference: {readability_diff:.2f} points")
    print("  *The Flesch Reading Ease score indicates how easy a text is to read. The difference shows the variation in reading complexity between the two answers.\n")
    
    print(f"ROUGE-1 F1 Score: {rouge['rouge1'].fmeasure:.2f} ({rouge['rouge1'].fmeasure*100:.0f}/100)")
    print("  *Measures overlap of unigrams (single words) between texts. A higher score indicates more overlap.\n")
    
    print(f"ROUGE-L F1 Score: {rouge['rougeL'].fmeasure:.2f} ({rouge['rougeL'].fmeasure*100:.0f}/100)")
    print("  *Considers the longest common subsequence, capturing fluency and sentence structure similarity.\n")
    
    # Overall score and improvement tips
    print("=== Overall Evaluation ===\n")
    print(f"Overall Score: {overall_score:.2f} ({overall_score*100:.0f}/100)")
    
    improvement_tips = generate_improvement_tips(overall_score)
    print("\nImprovement Tips:")
    print(improvement_tips)

# -------------------------------
# Example Usage
# -------------------------------
    