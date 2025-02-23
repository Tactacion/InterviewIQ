// app.js
import VideoProcessor from './video-processor.js';
import InterviewManager from './interview.js';

class App {
    constructor() {
        this.videoProcessor = new VideoProcessor();
        this.interviewManager = new InterviewManager(this.videoProcessor);
    }

    initialize() {
        // Initialize any global event listeners or settings
        document.addEventListener('DOMContentLoaded', () => {
            console.log('Application initialized');
            this.interviewManager.initialize();
        });

        // Handle global error catching
        window.onerror = (msg, url, lineNo, columnNo, error) => {
            console.error('Global error:', { msg, url, lineNo, columnNo, error });
            // You could add error reporting service here
        };
    }
}

// Initialize the application
const app = new App();
app.initialize();

export default App;