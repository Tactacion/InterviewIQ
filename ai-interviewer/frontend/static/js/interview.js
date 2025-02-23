class InterviewManager {
    constructor(videoProcessor) {
        this.videoProcessor = videoProcessor;
        this.sessionId = null;
        this.currentQuestionIndex = 0;
        this.questions = [];
        this.mediaStream = null;
        this.startTime = null;
        this.timerInterval = null;
        this.isRecording = false;
        this.messageQueue = [];
        this.isProcessingMessage = false;
    }

    initialize() {
        // Initialize DOM elements
        this.initializeElements();
        // Bind event listeners
        this.bindEventListeners();
    }

    initializeElements() {
        this.elements = {
            video: document.getElementById('videoElement'),
            startButton: document.getElementById('startButton'),
            endButton: document.getElementById('endButton'),
            prevButton: document.getElementById('prevQuestion'),
            nextButton: document.getElementById('nextQuestion'),
            chatInput: document.getElementById('chatInput'),
            sendButton: document.getElementById('sendMessage'),
            voiceButton: document.getElementById('voiceButton'),
            chatMessages: document.getElementById('chatMessages'),
            currentQuestion: document.getElementById('currentQuestion'),
            liveTimer: document.getElementById('liveTimer')
        };
    }

    bindEventListeners() {
        this.elements.startButton.addEventListener('click', () => this.startInterview());
        this.elements.endButton.addEventListener('click', () => this.endInterview());
        this.elements.prevButton.addEventListener('click', () => this.previousQuestion());
        this.elements.nextButton.addEventListener('click', () => this.nextQuestion());
        this.elements.sendButton.addEventListener('click', () => this.handleMessageSend());
        this.elements.voiceButton.addEventListener('click', () => this.startSpeechRecognition());
        this.elements.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleMessageSend();
        });
    }

    async startInterview() {
        const interviewType = document.getElementById('interviewType').value;
        if (!interviewType) {
            alert('Please select an interview type');
            return;
        }

        try {
            // Start camera
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
            this.elements.video.srcObject = this.mediaStream;

            // Initialize session
            const response = await fetch('/api/start_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: interviewType })
            });

            const data = await response.json();
            this.sessionId = data.session_id;
            this.questions = data.questions;
            
            this.showStage('camera');
            this.updateQuestion();
            this.startTimer();
            this.videoProcessor.startProcessing(
                this.elements.video,
                this.sessionId,
                (metrics) => this.updateMetrics(metrics)
            );
            this.isRecording = true;

        } catch (error) {
            console.error('Error starting interview:', error);
            alert('Failed to start interview. Please check console for details.');
        }
    }

    async handleMessageSend() {
        const message = this.elements.chatInput.value.trim();
        if (!message) return;

        this.addMessage(message, true);
        this.elements.chatInput.value = '';

        this.messageQueue.push(message);
        await this.processMessageQueue();
    }

    async processMessageQueue() {
        if (this.isProcessingMessage || this.messageQueue.length === 0) return;

        this.isProcessingMessage = true;
        const message = this.messageQueue.shift();

        try {
            const response = await fetch('/api/evaluate_answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    answer: message,
                    question_id: this.currentQuestionIndex
                })
            });

            const evaluation = await response.json();
            this.addMessage(evaluation.feedback, false);
        } catch (error) {
            console.error('Error evaluating answer:', error);
            this.addMessage('Failed to process response. Please try again.', false);
        }

        this.isProcessingMessage = false;
        if (this.messageQueue.length > 0) {
            await this.processMessageQueue();
        }
    }

    async startSpeechRecognition() {
        try {
            const response = await fetch('/get_message');
            const data = await response.json();
            
            if (data.message) {
                this.elements.chatInput.value = data.message;
                this.handleMessageSend();
            }
        } catch (error) {
            console.error('Error in speech recognition:', error);
        }
    }
    // In interview.js, update the updateMetrics method
    updateMetrics(metrics) {
        // Basic style for metric displays
        const metricStyle = 'padding: 8px; margin: 5px; font-weight: bold; font-size: 16px;';
        
        // Update smile intensity
        const smileElement = document.getElementById('smileIntensity');
        smileElement.textContent = `Smile: ${(metrics.smile_intensity * 100).toFixed(0)}%`;
        smileElement.style.cssText = metricStyle;
        smileElement.style.color = metrics.smile_intensity > 0.5 ? '#22c55e' : '#71717a';
    
        // Update eye contact
        const eyeElement = document.getElementById('eyeContact');
        eyeElement.textContent = `Eye Contact: ${metrics.eye_contact}`;
        eyeElement.style.cssText = metricStyle;
        eyeElement.style.color = metrics.eye_contact === 'Maintaining Eye Contact' ? '#22c55e' : '#71717a';
    
        // Update posture
        const postureElement = document.getElementById('posture');
        postureElement.textContent = `Posture: ${metrics.posture}`;
        postureElement.style.cssText = metricStyle;
        postureElement.style.color = metrics.posture === 'Upright Posture' ? '#22c55e' : '#71717a';
    
        // Update fidgeting
        const fidgetElement = document.getElementById('fidget');
        fidgetElement.textContent = `Fidgeting: ${metrics.fidget_status}`;
        fidgetElement.style.cssText = metricStyle;
        fidgetElement.style.color = metrics.fidget_status === 'No Fidgeting' ? '#22c55e' : '#71717a';
    }
    startTimer() {
        this.startTime = Date.now();
        this.timerInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const seconds = Math.floor(elapsed / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            
            this.elements.liveTimer.textContent = 
                `${String(hours).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
        }, 1000);
    }

    updateQuestion() {
        const question = this.questions[this.currentQuestionIndex];
        this.elements.currentQuestion.textContent = question;
        
        this.elements.prevButton.disabled = this.currentQuestionIndex === 0;
        this.elements.nextButton.disabled = this.currentQuestionIndex === this.questions.length - 1;
    }

    previousQuestion() {
        if (this.currentQuestionIndex > 0) {
            this.currentQuestionIndex--;
            this.updateQuestion();
        }
    }

    nextQuestion() {
        if (this.currentQuestionIndex < this.questions.length - 1) {
            this.currentQuestionIndex++;
            this.updateQuestion();
        }
    }

    addMessage(text, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${isUser ? 'message-sent' : 'message-received'}`;
        
        const timestamp = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        messageDiv.innerHTML = `
            ${text}
            <span class="message-timestamp">${timestamp}</span>
        `;
        
        this.elements.chatMessages.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
    }

    showStage(stageName) {
        document.querySelectorAll('.stage').forEach(stage => {
            stage.classList.remove('active');
        });
        document.getElementById(`${stageName}Stage`).classList.add('active');
    }

    async endInterview() {
        this.isRecording = false;
        
        // Stop recording
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
        }
        
        this.videoProcessor.stopProcessing();
        clearInterval(this.timerInterval);

        try {
            // Get chat messages
            const chatMessages = Array.from(this.elements.chatMessages.children).map(msg => ({
                text: msg.textContent,
                isUser: msg.classList.contains('message-sent')
            }));

            const response = await fetch('/api/end_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    session_id: this.sessionId,
                    interview_type: document.getElementById('interviewType').value,
                    duration: this.elements.liveTimer.textContent,
                    chat_history: chatMessages,
                    questions: this.questions,
                    questions_completed: this.currentQuestionIndex + 1
                })
            });

            const summary = await response.json();
            this.showReviewStage(summary);
        } catch (error) {
            console.error('Error ending session:', error);
        }
    }
    showReviewStage(summary) {
        this.showStage('review');
        
        // Update header colors
        document.getElementById('reviewType').style.color = 'white';
        document.getElementById('reviewDuration').style.color = 'white';
        
        const feedbackPoints = document.getElementById('feedbackPoints');
        feedbackPoints.innerHTML = '';
        
        // Create metrics section
        const metricsSection = document.createElement('div');
        metricsSection.className = 'metrics-section';
    
        // Add header
        const header = document.createElement('h3');
        header.className = 'metrics-header';
        header.textContent = 'Performance Metrics';
        metricsSection.appendChild(header);
    
        // Add overall score first
        const overallScore = document.createElement('div');
        overallScore.className = 'overall-score';
        overallScore.textContent = `Overall Performance Score: ${summary.overall_score.toFixed(1)}%`;
        metricsSection.appendChild(overallScore);
    
        // Add behavioral metrics
        const behavioralMetrics = [
            `Smile Engagement: ${(summary.metrics.smile_intensity * 100).toFixed(1)}%`,
            `Eye Contact: ${summary.metrics.eye_contact_percentage.toFixed(1)}%`,
            `Posture: ${summary.metrics.posture_percentage.toFixed(1)}%`,
            `Fidgeting Instances: ${summary.metrics.fidgeting_instances}`
        ];
    
        behavioralMetrics.forEach(metric => {
            const metricDiv = document.createElement('div');
            metricDiv.className = 'metric-item';
            metricDiv.textContent = metric;
            metricsSection.appendChild(metricDiv);
        });
    
        // Add analysis section
        const analysisSection = document.createElement('div');
        analysisSection.className = 'analysis-section';
        
        const analysisHeader = document.createElement('h3');
        analysisHeader.className = 'metrics-header';
        analysisHeader.textContent = 'Detailed Analysis';
        analysisSection.appendChild(analysisHeader);
    
        const analysisContent = document.createElement('div');
        analysisContent.style.fontWeight = '500';  // Make text semi-bold
        analysisContent.innerHTML = summary.analysis.replace(/\n/g, '<br>');
        analysisSection.appendChild(analysisContent);
    
        // Add sections to feedback points
        feedbackPoints.appendChild(metricsSection);
        feedbackPoints.appendChild(analysisSection);
    
        // Style the new interview button
        const newInterviewButton = document.getElementById('newInterviewButton');
        newInterviewButton.style.backgroundColor = '#60a5fa';
        newInterviewButton.style.color = 'white';
        newInterviewButton.style.fontWeight = '600';
        
        newInterviewButton.addEventListener('click', () => {
            location.reload();
        });
    }
}

export default InterviewManager;