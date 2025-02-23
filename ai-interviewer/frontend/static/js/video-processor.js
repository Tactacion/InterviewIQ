// video-processor.js
class VideoProcessor {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.processingInterval = null;
        this.isProcessing = false;
    }

    captureFrame(videoElement) {
        if (!videoElement || videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
            return null;
        }

        this.canvas.width = videoElement.videoWidth;
        this.canvas.height = videoElement.videoHeight;
        this.ctx.drawImage(videoElement, 0, 0);
        return this.canvas.toDataURL('image/jpeg', 0.8);
    }

    async processFrame(videoElement, sessionId) {
        try {
            const frame = this.captureFrame(videoElement);
            if (!frame) return null;

            const response = await fetch('/api/analyze_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    frame: frame
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error processing frame:', error);
            return null;
        }
    }

    startProcessing(videoElement, sessionId, onMetricsUpdate) {
        if (this.isProcessing) {
            this.stopProcessing();
        }

        this.isProcessing = true;
        this.processingInterval = setInterval(async () => {
            if (this.isProcessing && videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
                const metrics = await this.processFrame(videoElement, sessionId);
                if (metrics && onMetricsUpdate) {
                    onMetricsUpdate(metrics);
                }
            }
        }, 1000 / 30); // 30 fps
    }

    stopProcessing() {
        this.isProcessing = false;
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
    }
}

export default VideoProcessor;