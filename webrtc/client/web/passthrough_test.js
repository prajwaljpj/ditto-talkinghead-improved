/**
 * Audio Passthrough Test
 * Simple loopback - microphone audio → server → speaker output
 */

let ws = null;
let pc = null;

const elements = {
    serverUrl: document.getElementById('serverUrl'),
    micSelect: document.getElementById('micSelect'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    status: document.getElementById('status'),
    activeIndicator: document.getElementById('activeIndicator'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== Audio Passthrough Test Initialized ===');
    await loadMicrophones();
    showStatus('Ready to test', 'info');
});

async function loadMicrophones() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const microphones = devices.filter(device => device.kind === 'audioinput');

        elements.micSelect.innerHTML = '';
        microphones.forEach((mic, index) => {
            const option = document.createElement('option');
            option.value = mic.deviceId;
            option.text = mic.label || `Microphone ${index + 1}`;
            elements.micSelect.appendChild(option);
        });

        console.log(`Found ${microphones.length} microphones`);
    } catch (error) {
        console.error('Error loading microphones:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

async function startTest() {
    try {
        console.log('Connecting to server...');
        ws = new WebSocket(elements.serverUrl.value);

        ws.onopen = async () => {
            console.log('WebSocket connected');
            ws.send(JSON.stringify({ type: 'connect' }));
        };

        ws.onmessage = async (event) => {
            const message = JSON.parse(event.data);
            await handleMessage(message);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            showStatus('Connection error', 'error');
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            stopTest();
        };

        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        showStatus('Connecting...', 'info');

    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

async function handleMessage(message) {
    console.log('Received:', message.type);

    switch (message.type) {
        case 'ready':
            showStatus('Server ready, setting up audio...', 'info');
            await setupWebRTC();
            break;

        case 'answer':
            await pc.setRemoteDescription(new RTCSessionDescription({
                type: 'answer',
                sdp: message.sdp
            }));
            console.log('Answer set');
            break;

        case 'ice-candidate':
            await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
            break;

        case 'error':
            showStatus('Server error: ' + message.message, 'error');
            break;
    }
}

async function setupWebRTC() {
    pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.onicecandidate = (event) => {
        if (event.candidate) {
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

    pc.onconnectionstatechange = () => {
        console.log('Connection state:', pc.connectionState);
        if (pc.connectionState === 'connected') {
            showStatus('✅ Connected! Speak into your microphone!', 'success');
            elements.activeIndicator.classList.add('active');
        }
    };

    // Receive audio from server
    pc.ontrack = (event) => {
        console.log('Received track:', event.track.kind);

        if (event.streams && event.streams[0]) {
            // Play audio directly
            const audio = new Audio();
            audio.srcObject = event.streams[0];
            audio.play();

            console.log('✓ Audio playback started');
        }
    };

    // Get microphone audio
    const deviceId = elements.micSelect.value;
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            echoCancellation: false,  // Disable for pure passthrough
            noiseSuppression: false,
            autoGainControl: false,
            sampleRate: 48000
        }
    });

    // Add microphone track to peer connection
    stream.getAudioTracks().forEach(track => {
        pc.addTrack(track, stream);
        console.log('Added audio track to peer connection');
    });

    // Receive audio back
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // Create and send offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    ws.send(JSON.stringify({
        type: 'offer',
        sdp: pc.localDescription.sdp
    }));

    console.log('Offer sent');
}

function stopTest() {
    if (pc) {
        pc.close();
        pc = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.activeIndicator.classList.remove('active');
    showStatus('Test stopped', 'info');
}

function showStatus(message, type) {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
    elements.status.style.display = 'block';
    console.log(`[${type.toUpperCase()}] ${message}`);
}

elements.startBtn.addEventListener('click', startTest);
elements.stopBtn.addEventListener('click', stopTest);
