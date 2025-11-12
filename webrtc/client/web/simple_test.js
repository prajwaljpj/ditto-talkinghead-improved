/**
 * Simple Ditto Audio Test
 * Just plays the remote stream directly - no recording complexity
 */

let mediaRecorder = null;
let recordedAudioBlob = null;
let recordingStartTime = null;
let recordingInterval = null;
let ws = null;
let pc = null;

const elements = {
    serverUrl: document.getElementById('serverUrl'),
    avatarSource: document.getElementById('avatarSource'),
    micSelect: document.getElementById('micSelect'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    status: document.getElementById('status'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    recordingTime: document.getElementById('recordingTime'),
    resultVideo: document.getElementById('resultVideo'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== Simple Audio Test Initialized ===');
    await loadMicrophones();
    showStatus('Ready to record', 'info');
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

async function startRecording() {
    try {
        const deviceId = elements.micSelect.value;

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });

        mediaRecorder = new MediaRecorder(stream);
        const recordedChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            recordedAudioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            console.log(`Recorded: ${(recordedAudioBlob.size / 1024).toFixed(2)} KB`);

            showStatus('Processing audio through Ditto...', 'info');
            await processAudio();
        };

        mediaRecorder.start();

        recordingStartTime = Date.now();
        updateRecordingTime();
        recordingInterval = setInterval(updateRecordingTime, 100);
        elements.recordingIndicator.classList.add('active');
        elements.startBtn.disabled = true;
        elements.stopBtn.disabled = false;
        showStatus('Recording... Speak now!', 'error');

        console.log('Recording started');
    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());

        clearInterval(recordingInterval);
        elements.recordingIndicator.classList.remove('active');
        elements.startBtn.disabled = false;
        elements.stopBtn.disabled = true;
    }
}

function updateRecordingTime() {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    elements.recordingTime.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

async function processAudio() {
    try {
        console.log('Connecting to server...');
        ws = new WebSocket(elements.serverUrl.value);

        ws.onopen = async () => {
            console.log('WebSocket connected');
            ws.send(JSON.stringify({
                type: 'connect',
                source: elements.avatarSource.value
            }));
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
        };

    } catch (error) {
        console.error('Error:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

async function handleMessage(message) {
    console.log('Received:', message.type);

    switch (message.type) {
        case 'ready':
            showStatus('Server ready, connecting...', 'info');
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
            showStatus('✅ Connected! Playing video...', 'success');
        }
    };

    // THIS IS THE KEY: Just set srcObject directly!
    pc.ontrack = (event) => {
        console.log('Received track:', event.track.kind);

        if (event.track.kind === 'audio') {
            const settings = event.track.getSettings();
            console.log('=== RECEIVED AUDIO TRACK ===');
            console.log('Sample rate:', settings.sampleRate, 'Hz');
            console.log('Channels:', settings.channelCount);
            console.log('Settings:', settings);
        }

        if (event.streams && event.streams[0]) {
            // Just play the stream directly - no recording!
            elements.resultVideo.srcObject = event.streams[0];
            elements.resultVideo.muted = false;
            elements.resultVideo.volume = 1.0;

            console.log('✓ Stream connected to video element');
            showStatus('✅ Playing! Listen to the audio quality!', 'success');
        }
    };

    // Add audio track from recording
    const audioTrack = await createAudioTrack(recordedAudioBlob);

    const trackSettings = audioTrack.getSettings();
    console.log('=== SENDING AUDIO TRACK ===');
    console.log('Track sample rate:', trackSettings.sampleRate, 'Hz');
    console.log('Track settings:', trackSettings);

    pc.addTrack(audioTrack);

    // Receive video
    pc.addTransceiver('video', { direction: 'recvonly' });

    // Create and send offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    ws.send(JSON.stringify({
        type: 'offer',
        sdp: pc.localDescription.sdp
    }));

    console.log('Offer sent');
}

async function createAudioTrack(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Log sample rates to detect mismatch
    console.log('=== SAMPLE RATE CHECK ===');
    console.log('Requested AudioContext rate:', 16000);
    console.log('Actual AudioContext rate:', audioContext.sampleRate);
    console.log('Decoded audio buffer rate:', audioBuffer.sampleRate);
    console.log('Duration:', audioBuffer.duration, 'seconds');

    if (audioContext.sampleRate !== audioBuffer.sampleRate) {
        console.error('❌ SAMPLE RATE MISMATCH! This will cause pitch issues!');
        console.error(`Context: ${audioContext.sampleRate}Hz, Buffer: ${audioBuffer.sampleRate}Hz`);
    }

    const destination = audioContext.createMediaStreamDestination();

    // Play the audio once (no looping)
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(destination);
    source.start();

    console.log('Audio track created (plays once)');
    console.log('Output track sample rate:', destination.stream.getAudioTracks()[0].getSettings().sampleRate);

    return destination.stream.getAudioTracks()[0];
}

function showStatus(message, type) {
    elements.status.textContent = message;
    elements.status.className = `status ${type}`;
    elements.status.style.display = 'block';
    console.log(`[${type.toUpperCase()}] ${message}`);
}

elements.startBtn.addEventListener('click', startRecording);
elements.stopBtn.addEventListener('click', stopRecording);
