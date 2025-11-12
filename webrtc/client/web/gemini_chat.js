/**
 * Gemini + Ditto Conversational Avatar Client
 *
 * Handles WebRTC connection, audio streaming, and UI updates
 * for full-duplex conversational AI avatar.
 */

class GeminiDittoClient {
    constructor() {
        // WebRTC
        this.pc = null;
        this.ws = null;
        this.localStream = null;

        // State
        this.isConnected = false;
        this.isMuted = false;
        this.turnCount = 0;
        this.interruptCount = 0;
        this.sessionStartTime = null;
        this.conversationHistory = [];

        // Audio analysis
        this.audioContext = null;
        this.micAnalyser = null;
        this.avatarAnalyser = null;

        // Elements
        this.remoteVideo = document.getElementById('remoteVideo');
        this.statusBadge = document.getElementById('statusBadge');
        this.statusText = document.getElementById('statusText');
        this.emotionBadge = document.getElementById('emotionBadge');
        this.emotionIcon = document.getElementById('emotionIcon');
        this.emotionText = document.getElementById('emotionText');
        this.transcript = document.getElementById('transcript');

        // Control buttons
        this.connectBtn = document.getElementById('connectBtn');
        this.disconnectBtn = document.getElementById('disconnectBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.muteBtn = document.getElementById('muteBtn');
        this.clearTranscriptBtn = document.getElementById('clearTranscriptBtn');

        // Settings
        this.serverUrl = document.getElementById('serverUrl');
        this.vadThreshold = document.getElementById('vadThreshold');
        this.vadThresholdValue = document.getElementById('vadThresholdValue');
        this.emotionSensitivity = document.getElementById('emotionSensitivity');
        this.enableInterruptions = document.getElementById('enableInterruptions');

        // Stats
        this.turnCountEl = document.getElementById('turnCount');
        this.durationEl = document.getElementById('duration');
        this.latencyEl = document.getElementById('latency');
        this.interruptCountEl = document.getElementById('interruptCount');
        this.micLevel = document.getElementById('micLevel');
        this.avatarLevel = document.getElementById('avatarLevel');

        // Bind event listeners
        this.setupEventListeners();

        // Start duration counter
        setInterval(() => this.updateDuration(), 1000);

        // Start audio level monitoring
        this.startAudioLevelMonitoring();
    }

    setupEventListeners() {
        // Controls
        this.connectBtn.addEventListener('click', () => this.connect());
        this.disconnectBtn.addEventListener('click', () => this.disconnect());
        this.resetBtn.addEventListener('click', () => this.resetConversation());
        this.muteBtn.addEventListener('click', () => this.toggleMute());
        this.clearTranscriptBtn.addEventListener('click', () => this.clearTranscript());

        // Settings
        this.vadThreshold.addEventListener('input', (e) => {
            this.vadThresholdValue.textContent = e.target.value;
        });
    }

    async connect() {
        try {
            this.updateStatus('connecting', 'Connecting...');
            this.connectBtn.disabled = true;

            // Check if mediaDevices is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error(
                    'WebRTC not available. Please use HTTPS or access via localhost (not 0.0.0.0). ' +
                    'Current URL: ' + window.location.href
                );
            }

            // Get user media (microphone)
            console.log('Requesting microphone access...');
            this.localStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 48000
                },
                video: false
            });

            console.log('Microphone access granted');

            // Setup audio analysis
            this.setupAudioAnalysis();

            // Create peer connection
            this.pc = new RTCPeerConnection({
                iceServers: [
                    { urls: 'stun:stun.l.google.com:19302' }
                ]
            });

            // Add local audio track
            this.localStream.getAudioTracks().forEach(track => {
                this.pc.addTrack(track, this.localStream);
                console.log('Added local audio track');
            });

            // Handle remote stream
            this.pc.ontrack = (event) => {
                console.log('Received remote track:', event.track.kind);
                if (event.track.kind === 'video') {
                    this.remoteVideo.srcObject = event.streams[0];
                    console.log('Remote video stream connected');
                }
            };

            // Handle ICE candidates
            this.pc.onicecandidate = (event) => {
                if (event.candidate) {
                    this.ws.send(JSON.stringify({
                        type: 'ice',
                        candidate: event.candidate
                    }));
                }
            };

            // Handle connection state changes
            this.pc.onconnectionstatechange = () => {
                console.log('Connection state:', this.pc.connectionState);
                if (this.pc.connectionState === 'connected') {
                    this.onConnected();
                } else if (this.pc.connectionState === 'disconnected' ||
                           this.pc.connectionState === 'failed') {
                    this.onDisconnected();
                }
            };

            // Connect to signaling server
            const wsUrl = this.serverUrl.value;
            console.log('Connecting to signaling server:', wsUrl);
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = async () => {
                console.log('WebSocket connected');

                // Create and send offer
                const offer = await this.pc.createOffer({
                    offerToReceiveAudio: false,
                    offerToReceiveVideo: true
                });

                await this.pc.setLocalDescription(offer);

                this.ws.send(JSON.stringify({
                    type: 'offer',
                    sdp: offer.sdp
                }));

                console.log('Sent offer to server');
            };

            this.ws.onmessage = async (event) => {
                const message = JSON.parse(event.data);

                if (message.type === 'answer') {
                    console.log('Received answer from server');
                    await this.pc.setRemoteDescription({
                        type: 'answer',
                        sdp: message.sdp
                    });
                } else if (message.type === 'ice') {
                    console.log('Received ICE candidate');
                    await this.pc.addIceCandidate(message.candidate);
                } else if (message.type === 'transcript') {
                    // Handle transcript updates
                    this.addTranscriptItem(message.role, message.text, message.emotion);
                } else if (message.type === 'emotion') {
                    // Handle emotion updates
                    this.updateEmotion(message.emotion);
                } else if (message.type === 'stats') {
                    // Handle stats updates
                    this.updateStats(message.stats);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('disconnected', 'Connection Error');
                this.connectBtn.disabled = false;
            };

            this.ws.onclose = () => {
                console.log('WebSocket closed');
                this.onDisconnected();
            };

        } catch (error) {
            console.error('Connection error:', error);
            alert('Failed to connect: ' + error.message);
            this.updateStatus('disconnected', 'Disconnected');
            this.connectBtn.disabled = false;
        }
    }

    disconnect() {
        console.log('Disconnecting...');

        // Close peer connection
        if (this.pc) {
            this.pc.close();
            this.pc = null;
        }

        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        // Stop local stream
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.localStream = null;
        }

        // Stop audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.onDisconnected();
    }

    onConnected() {
        console.log('Successfully connected!');
        this.isConnected = true;
        this.sessionStartTime = Date.now();
        this.updateStatus('connected', 'Connected');
        this.connectBtn.disabled = true;
        this.disconnectBtn.disabled = false;
        this.resetBtn.disabled = false;
        this.muteBtn.disabled = false;
    }

    onDisconnected() {
        this.isConnected = false;
        this.updateStatus('disconnected', 'Disconnected');
        this.connectBtn.disabled = false;
        this.disconnectBtn.disabled = true;
        this.resetBtn.disabled = true;
        this.muteBtn.disabled = true;
    }

    toggleMute() {
        this.isMuted = !this.isMuted;

        if (this.localStream) {
            this.localStream.getAudioTracks().forEach(track => {
                track.enabled = !this.isMuted;
            });
        }

        const muteIcon = document.getElementById('muteIcon');
        const muteText = document.getElementById('muteText');

        if (this.isMuted) {
            muteIcon.textContent = '🔇';
            muteText.textContent = 'Unmute';
            this.muteBtn.classList.remove('btn-secondary');
            this.muteBtn.classList.add('btn-danger');
        } else {
            muteIcon.textContent = '🎤';
            muteText.textContent = 'Mute';
            this.muteBtn.classList.remove('btn-danger');
            this.muteBtn.classList.add('btn-secondary');
        }
    }

    resetConversation() {
        if (!confirm('Are you sure you want to reset the conversation?')) {
            return;
        }

        // Send reset command
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'reset' }));
        }

        // Reset local state
        this.turnCount = 0;
        this.interruptCount = 0;
        this.conversationHistory = [];
        this.clearTranscript();
        this.updateStats({});
        this.updateEmotion('neutral');
    }

    clearTranscript() {
        this.transcript.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">💬</div>
                <p>Start a conversation to see transcript here</p>
            </div>
        `;
    }

    addTranscriptItem(role, text, emotion = 'neutral') {
        // Remove empty state if present
        const emptyState = this.transcript.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        // Create transcript item
        const item = document.createElement('div');
        item.className = `transcript-item ${role}`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });

        const emotionEmoji = this.getEmotionEmoji(emotion);

        item.innerHTML = `
            <div class="transcript-header">
                <span class="transcript-role">
                    ${role === 'user' ? 'You' : 'Avatar'}
                    ${role === 'model' ? `<span class="transcript-emotion">${emotionEmoji} ${emotion}</span>` : ''}
                </span>
                <span class="transcript-time">${timeStr}</span>
            </div>
            <div class="transcript-text">${text}</div>
        `;

        this.transcript.appendChild(item);

        // Auto-scroll to bottom
        this.transcript.scrollTop = this.transcript.scrollHeight;

        // Update turn count
        if (role === 'model') {
            this.turnCount++;
            this.turnCountEl.textContent = this.turnCount;
        }

        // Store in history
        this.conversationHistory.push({ role, text, emotion, time: now });
    }

    updateEmotion(emotion) {
        const emotionMap = {
            'neutral': { icon: '😐', label: 'Neutral' },
            'neu': { icon: '😐', label: 'Neutral' },
            'happy': { icon: '😊', label: 'Happy' },
            'hap': { icon: '😊', label: 'Happy' },
            'sad': { icon: '😢', label: 'Sad' },
            'angry': { icon: '😠', label: 'Angry' },
            'ang': { icon: '😠', label: 'Angry' },
            'surprised': { icon: '😲', label: 'Surprised' },
            'sur': { icon: '😲', label: 'Surprised' },
            'fear': { icon: '😨', label: 'Fear' },
            'disgusted': { icon: '🤢', label: 'Disgusted' },
            'contemptuous': { icon: '😤', label: 'Contemptuous' }
        };

        const emo = emotionMap[emotion] || emotionMap['neutral'];
        this.emotionIcon.textContent = emo.icon;
        this.emotionText.textContent = emo.label;
    }

    getEmotionEmoji(emotion) {
        const emojiMap = {
            'neutral': '😐', 'neu': '😐',
            'happy': '😊', 'hap': '😊',
            'sad': '😢',
            'angry': '😠', 'ang': '😠',
            'surprised': '😲', 'sur': '😲',
            'fear': '😨',
            'disgusted': '🤢',
            'contemptuous': '😤'
        };
        return emojiMap[emotion] || '😐';
    }

    updateStatus(state, text) {
        this.statusBadge.className = `status-badge ${state}`;
        this.statusText.textContent = text;
    }

    updateDuration() {
        if (!this.sessionStartTime) {
            this.durationEl.textContent = '0:00';
            return;
        }

        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        this.durationEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    updateStats(stats) {
        if (stats.latency !== undefined) {
            this.latencyEl.textContent = `${stats.latency}ms`;
        }
        if (stats.interruptions !== undefined) {
            this.interruptCount = stats.interruptions;
            this.interruptCountEl.textContent = stats.interruptions;
        }
    }

    setupAudioAnalysis() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

            // Microphone analysis
            const micSource = this.audioContext.createMediaStreamSource(this.localStream);
            this.micAnalyser = this.audioContext.createAnalyser();
            this.micAnalyser.fftSize = 256;
            micSource.connect(this.micAnalyser);

            console.log('Audio analysis setup complete');
        } catch (error) {
            console.error('Error setting up audio analysis:', error);
        }
    }

    startAudioLevelMonitoring() {
        const updateLevels = () => {
            // Update microphone level
            if (this.micAnalyser) {
                const dataArray = new Uint8Array(this.micAnalyser.frequencyBinCount);
                this.micAnalyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                const level = Math.min(100, (average / 255) * 150);
                this.micLevel.style.width = `${level}%`;
            }

            // Avatar level would come from remote audio track
            // For now, just show activity during conversation
            if (this.isConnected && this.remoteVideo.srcObject) {
                const randomLevel = Math.random() * 50 + 10;
                this.avatarLevel.style.width = `${randomLevel}%`;
            } else {
                this.avatarLevel.style.width = '0%';
            }

            requestAnimationFrame(updateLevels);
        };

        updateLevels();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const client = new GeminiDittoClient();
    console.log('Gemini Ditto Client initialized');
});
