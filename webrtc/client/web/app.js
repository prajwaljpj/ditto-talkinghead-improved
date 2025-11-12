/**
 * Ditto Talking Head WebRTC Client
 *
 * Handles WebRTC connection to the Ditto avatar server:
 * - WebSocket signaling
 * - Audio input capture
 * - Video stream display
 */

// Global state
let ws = null;
let pc = null;
let localStream = null;
let statsInterval = null;
let audioSender = null;  // Keep reference to audio sender

// DOM elements
const elements = {
    serverUrl: document.getElementById('serverUrl'),
    avatarSource: document.getElementById('avatarSource'),
    connectBtn: document.getElementById('connectBtn'),
    disconnectBtn: document.getElementById('disconnectBtn'),
    startAudioBtn: document.getElementById('startAudioBtn'),
    stopAudioBtn: document.getElementById('stopAudioBtn'),
    micSelect: document.getElementById('micSelect'),
    remoteVideo: document.getElementById('remoteVideo'),
    status: document.getElementById('status'),
    stats: document.getElementById('stats'),
    connectionState: document.getElementById('connectionState'),
    iceState: document.getElementById('iceState'),
    videoStats: document.getElementById('videoStats'),
    audioStats: document.getElementById('audioStats'),
    audioLevel: document.getElementById('audioLevel'),
    audioLevelBar: document.getElementById('audioLevelBar'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== WebRTC Support Check ===');
    console.log('navigator.mediaDevices:', navigator.mediaDevices);
    console.log('getUserMedia:', navigator.mediaDevices ? navigator.mediaDevices.getUserMedia : 'N/A');
    console.log('Secure context:', window.isSecureContext);
    console.log('Location:', window.location.href);
    console.log('Protocol:', window.location.protocol);
    console.log('============================');

    // Check for WebRTC support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMsg = `
            Your browser does not support WebRTC or you're not in a secure context.

            Current URL: ${window.location.href}
            Secure context: ${window.isSecureContext}

            Solution: Access via http://localhost:8000 (NOT 0.0.0.0 or IP address)
        `;
        showStatus(errorMsg, 'error');
        elements.connectBtn.disabled = true;
        console.error('WebRTC not supported:', {
            mediaDevices: !!navigator.mediaDevices,
            getUserMedia: navigator.mediaDevices ? !!navigator.mediaDevices.getUserMedia : false,
            isSecureContext: window.isSecureContext,
            protocol: window.location.protocol,
            hostname: window.location.hostname
        });
        return;
    }

    console.log('✓ WebRTC is supported');
    await loadMicrophones();
    showStatus('Ready to connect', 'info');
});

/**
 * Load available microphones
 */
async function loadMicrophones() {
    try {
        // First request permission to get labeled devices
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
        } catch (permError) {
            console.log('Microphone permission not granted yet');
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const microphones = devices.filter(device => device.kind === 'audioinput');

        elements.micSelect.innerHTML = '';
        microphones.forEach((mic, index) => {
            const option = document.createElement('option');
            option.value = mic.deviceId;
            option.text = mic.label || `Microphone ${index + 1}`;
            elements.micSelect.appendChild(option);
        });

        if (microphones.length === 0) {
            showStatus('No microphones found. Please connect a microphone.', 'error');
        }
    } catch (error) {
        console.error('Error loading microphones:', error);
        showStatus('Error loading microphones: ' + error.message, 'error');
    }
}

/**
 * Connect to avatar server
 */
async function connect() {
    const serverUrl = elements.serverUrl.value;
    const avatarSource = elements.avatarSource.value;

    if (!serverUrl || !avatarSource) {
        showStatus('Please enter server URL and avatar source', 'error');
        return;
    }

    try {
        showStatus('Connecting to server...', 'info');
        elements.connectBtn.disabled = true;

        // Create WebSocket connection
        ws = new WebSocket(serverUrl);

        ws.onopen = () => {
            console.log('WebSocket connected');
            showStatus('Connected to server. Initializing avatar...', 'info');

            // Send connect message
            ws.send(JSON.stringify({
                type: 'connect',
                source: avatarSource
            }));
        };

        ws.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            await handleSignalingMessage(message);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            showStatus('WebSocket error', 'error');
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            showStatus('Disconnected from server', 'error');
            disconnect();
        };

    } catch (error) {
        console.error('Connection error:', error);
        showStatus('Connection error: ' + error.message, 'error');
        elements.connectBtn.disabled = false;
    }
}

/**
 * Handle signaling messages from server
 */
async function handleSignalingMessage(message) {
    console.log('Received message:', message);

    switch (message.type) {
        case 'ready':
            showStatus('Server ready. Creating WebRTC connection...', 'info');
            await createPeerConnection();
            await createOffer();
            break;

        case 'offer':
            // Server is initiating renegotiation (e.g., for audio echo)
            await handleOffer(message.sdp);
            break;

        case 'answer':
            await handleAnswer(message.sdp);
            break;

        case 'ice-candidate':
            await handleIceCandidate(message.candidate);
            break;

        case 'error':
            showStatus('Server error: ' + message.message, 'error');
            break;

        default:
            console.log('Unknown message type:', message.type);
    }
}

/**
 * Create WebRTC peer connection
 */
async function createPeerConnection() {
    // STUN/TURN configuration
    const configuration = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
        ]
    };

    pc = new RTCPeerConnection(configuration);

    // Handle ICE candidates
    pc.onicecandidate = (event) => {
        if (event.candidate) {
            console.log('Sending ICE candidate');
            ws.send(JSON.stringify({
                type: 'ice-candidate',
                candidate: {
                    candidate: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex
                }
            }));
        }
    };

    // Handle connection state changes
    pc.onconnectionstatechange = () => {
        console.log('Connection state:', pc.connectionState);
        elements.connectionState.textContent = `State: ${pc.connectionState}`;

        if (pc.connectionState === 'connected') {
            showStatus('WebRTC connected! You can now start speaking.', 'success');
            elements.startAudioBtn.disabled = false;
            elements.disconnectBtn.style.display = 'block';
            elements.connectBtn.style.display = 'none';
            elements.stats.style.display = 'block';
            startStatsMonitoring();
        } else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
            showStatus('Connection failed or disconnected', 'error');
            disconnect();
        }
    };

    // Handle ICE connection state
    pc.oniceconnectionstatechange = () => {
        console.log('ICE state:', pc.iceConnectionState);
        elements.iceState.textContent = `ICE: ${pc.iceConnectionState}`;
    };

    // Handle incoming media streams (video + audio)
    pc.ontrack = (event) => {
        console.log('=== Received remote track ===');
        console.log('Track kind:', event.track.kind);
        console.log('Track ID:', event.track.id);
        console.log('Track label:', event.track.label);
        console.log('Track readyState:', event.track.readyState);
        console.log('Track muted:', event.track.muted);
        console.log('Track enabled:', event.track.enabled);
        console.log('Streams:', event.streams.length);

        // Set the stream when we receive any track (video or audio)
        // The MediaStream will contain all tracks (video + audio)
        if (event.streams && event.streams[0]) {
            const stream = event.streams[0];

            // Always update srcObject to ensure it has latest tracks
            elements.remoteVideo.srcObject = stream;

            // Log track information FIRST
            const videoTracks = stream.getVideoTracks();
            const audioTracks = stream.getAudioTracks();

            // CRITICAL: Ensure video element can play audio
            elements.remoteVideo.muted = false;
            elements.remoteVideo.volume = 1.0;

            // Force the video element to have audio enabled
            if (audioTracks.length > 0) {
                audioTracks.forEach(track => {
                    track.enabled = true;
                    console.log('Enabled audio track:', track.id);
                });
            }

            // Try to play (in case autoplay is blocked)
            elements.remoteVideo.play().then(() => {
                console.log('✓ Video element playing successfully');
                console.log('  muted:', elements.remoteVideo.muted);
                console.log('  volume:', elements.remoteVideo.volume);
                console.log('  paused:', elements.remoteVideo.paused);
            }).catch(e => {
                console.warn('Autoplay blocked, user interaction may be needed:', e);
            });
            console.log('Remote stream updated:');
            console.log('  Video tracks:', videoTracks.length, videoTracks.map(t => `${t.label} [${t.readyState}]`));
            console.log('  Audio tracks:', audioTracks.length, audioTracks.map(t => `${t.label} [${t.readyState}, muted=${t.muted}, enabled=${t.enabled}]`));

            if (audioTracks.length > 0) {
                console.log('✓ Audio track received! Checking if it will play...');

                // Check audio track state
                audioTracks.forEach((track, i) => {
                    console.log(`  Audio track ${i}:`, {
                        readyState: track.readyState,
                        muted: track.muted,
                        enabled: track.enabled,
                        id: track.id
                    });
                });
            } else {
                console.warn('⚠ No audio track in stream. Audio echo may not be working.');
            }
        }
    };

    console.log('Peer connection created');
}

/**
 * Create and send SDP offer
 */
async function createOffer() {
    try {
        // Add transceiver for receiving video
        pc.addTransceiver('video', { direction: 'recvonly' });

        // Add transceiver for bidirectional audio (even if not sending yet)
        // This sets up the connection to support audio from the start
        const audioTransceiver = pc.addTransceiver('audio', { direction: 'sendrecv' });
        audioSender = audioTransceiver.sender;
        console.log('Audio transceiver added to initial offer');

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        console.log('Sending offer (with audio support)');
        ws.send(JSON.stringify({
            type: 'offer',
            sdp: pc.localDescription.sdp
        }));

    } catch (error) {
        console.error('Error creating offer:', error);
        showStatus('Error creating offer: ' + error.message, 'error');
    }
}

/**
 * Handle SDP offer from server (renegotiation)
 */
async function handleOffer(sdp) {
    try {
        console.log('Received offer from server (renegotiation)');

        const offer = new RTCSessionDescription({
            type: 'offer',
            sdp: sdp
        });

        await pc.setRemoteDescription(offer);
        console.log('Offer received and set');

        // Create answer
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        // Send answer back to server
        ws.send(JSON.stringify({
            type: 'answer',
            sdp: pc.localDescription.sdp
        }));

        console.log('Answer sent to server');

    } catch (error) {
        console.error('Error handling offer:', error);
        showStatus('Error handling offer: ' + error.message, 'error');
    }
}

/**
 * Handle SDP answer from server
 */
async function handleAnswer(sdp) {
    try {
        const answer = new RTCSessionDescription({
            type: 'answer',
            sdp: sdp
        });

        await pc.setRemoteDescription(answer);
        console.log('Answer received and set');

    } catch (error) {
        console.error('Error handling answer:', error);
        showStatus('Error handling answer: ' + error.message, 'error');
    }
}

/**
 * Handle ICE candidate from server
 */
async function handleIceCandidate(candidate) {
    try {
        if (candidate) {
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
            console.log('ICE candidate added');
        }
    } catch (error) {
        console.error('Error adding ICE candidate:', error);
    }
}

/**
 * Start audio input
 */
async function startAudio() {
    try {
        const deviceId = elements.micSelect.value;

        // Get microphone stream
        localStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000  // Ditto expects 16kHz
            }
        });

        // Replace the track on the existing audio sender (no renegotiation needed!)
        const audioTrack = localStream.getAudioTracks()[0];

        if (audioSender) {
            await audioSender.replaceTrack(audioTrack);
            console.log('Audio track replaced on existing sender');
        } else {
            console.error('No audio sender available!');
            showStatus('Error: Audio not set up properly. Please reconnect.', 'error');
            return;
        }

        // Show audio level indicator
        elements.audioLevel.style.display = 'block';
        startAudioLevelMonitoring();

        elements.startAudioBtn.style.display = 'none';
        elements.stopAudioBtn.style.display = 'block';

        showStatus('Microphone active. Speak to animate avatar! (Audio will be echoed back)', 'success');

    } catch (error) {
        console.error('Error starting audio:', error);
        showStatus('Error accessing microphone: ' + error.message, 'error');
    }
}

/**
 * Stop audio input
 * NOTE: This only stops the microphone input. The video stream continues playing.
 */
function stopAudio() {
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }

    // Replace the audio track with null (stops sending audio to server)
    if (audioSender) {
        audioSender.replaceTrack(null).then(() => {
            console.log('Audio input stopped, but video playback continues');
        }).catch(err => {
            console.error('Error stopping audio track:', err);
        });
    }

    elements.audioLevel.style.display = 'none';
    elements.startAudioBtn.style.display = 'block';
    elements.stopAudioBtn.style.display = 'none';

    showStatus('Microphone stopped. Video continues playing.', 'info');
}

/**
 * Monitor audio level
 */
function startAudioLevelMonitoring() {
    const audioContext = new AudioContext();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(localStream);
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    microphone.connect(analyser);
    analyser.fftSize = 256;

    function updateLevel() {
        if (!localStream) return;

        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        const percentage = Math.min(100, (average / 128) * 100);

        elements.audioLevelBar.style.width = percentage + '%';

        requestAnimationFrame(updateLevel);
    }

    updateLevel();
}

/**
 * Monitor connection statistics
 */
function startStatsMonitoring() {
    statsInterval = setInterval(async () => {
        if (!pc) return;

        try {
            const stats = await pc.getStats();
            let videoStats = {};
            let audioStats = {};

            stats.forEach(stat => {
                if (stat.type === 'inbound-rtp' && stat.kind === 'video') {
                    videoStats = {
                        fps: stat.framesPerSecond || 0,
                        frames: stat.framesReceived || 0,
                        bytes: (stat.bytesReceived / 1024 / 1024).toFixed(2) + ' MB'
                    };
                } else if (stat.type === 'inbound-rtp' && stat.kind === 'audio') {
                    audioStats = {
                        packets: stat.packetsReceived || 0,
                        bytes: (stat.bytesReceived / 1024).toFixed(2) + ' KB'
                    };
                }
            });

            if (videoStats.fps !== undefined) {
                elements.videoStats.textContent = `Video: ${videoStats.fps} fps, ${videoStats.frames} frames, ${videoStats.bytes}`;
            }

            if (audioStats.packets !== undefined) {
                elements.audioStats.textContent = `Audio: ${audioStats.packets} packets, ${audioStats.bytes}`;
            }

        } catch (error) {
            console.error('Error getting stats:', error);
        }
    }, 1000);
}

/**
 * Disconnect from server
 */
function disconnect() {
    stopAudio();

    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }

    if (pc) {
        pc.close();
        pc = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    // Reset audio sender reference
    audioSender = null;

    elements.remoteVideo.srcObject = null;
    elements.stats.style.display = 'none';
    elements.connectBtn.style.display = 'block';
    elements.connectBtn.disabled = false;
    elements.disconnectBtn.style.display = 'none';
    elements.startAudioBtn.disabled = true;

    showStatus('Disconnected', 'info');
}

/**
 * Show status message
 */
function showStatus(message, type) {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
    elements.status.style.display = 'block';
    console.log(`[${type.toUpperCase()}] ${message}`);
}
