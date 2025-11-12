/**
 * Ditto Audio Recording Test
 * Records audio, sends to server, receives video, plays back result
 */

// Global state
let mediaRecorder = null;
let recordedChunks = [];
let audioContext = null;
let analyser = null;
let microphone = null;
let recordingStartTime = null;
let recordingInterval = null;
let ws = null;
let pc = null;
let recordedAudioBlob = null;
let processStartTime = null;
let receivedVideoBlob = null;

// DOM elements
const elements = {
    serverUrl: document.getElementById('serverUrl'),
    avatarSource: document.getElementById('avatarSource'),
    micSelect: document.getElementById('micSelect'),
    startRecordBtn: document.getElementById('startRecordBtn'),
    stopRecordBtn: document.getElementById('stopRecordBtn'),
    playbackBtn: document.getElementById('playbackBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    status: document.getElementById('status'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    recordingTime: document.getElementById('recordingTime'),
    audioVisualizer: document.getElementById('audioVisualizer'),
    resultVideo: document.getElementById('resultVideo'),
    stats: document.getElementById('stats'),
    statDuration: document.getElementById('statDuration'),
    statAudioSize: document.getElementById('statAudioSize'),
    statVideoSize: document.getElementById('statVideoSize'),
    statProcessTime: document.getElementById('statProcessTime'),
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    console.log('=== Audio Recording Test Initialized ===');

    // Check codec support
    console.log('=== Codec Support ===');
    const codecs = [
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=vp9,opus',
        'video/webm',
        'video/mp4'
    ];
    codecs.forEach(codec => {
        console.log(`${codec}: ${MediaRecorder.isTypeSupported(codec) ? '✓' : '✗'}`);
    });
    console.log('====================');

    await loadMicrophones();
    setupVisualizerCanvas();
    showStatus('Ready to record', 'info');
});

/**
 * Load available microphones
 */
async function loadMicrophones() {
    try {
        // Request permission first
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

/**
 * Setup audio visualizer canvas
 */
function setupVisualizerCanvas() {
    const canvas = elements.audioVisualizer;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

/**
 * Start recording
 */
async function startRecording() {
    try {
        const deviceId = elements.micSelect.value;

        // Get microphone stream
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                deviceId: deviceId ? { exact: deviceId } : undefined,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000
            }
        });

        // Setup MediaRecorder
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm;codecs=opus'
        });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            console.log('Recording stopped, processing...');
            recordedAudioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            console.log(`Recorded audio: ${(recordedAudioBlob.size / 1024).toFixed(2)} KB`);

            // Update stats
            const duration = (Date.now() - recordingStartTime) / 1000;
            elements.statDuration.textContent = `${duration.toFixed(1)}s`;
            elements.statAudioSize.textContent = `${(recordedAudioBlob.size / 1024).toFixed(2)} KB`;
            elements.stats.classList.add('show');

            // Process through Ditto
            await processAudioThroughDitto();
        };

        mediaRecorder.start(100); // Collect data every 100ms

        // Setup audio visualizer
        setupVisualizer(stream);

        // Update UI
        recordingStartTime = Date.now();
        updateRecordingTime();
        recordingInterval = setInterval(updateRecordingTime, 100);
        elements.recordingIndicator.classList.add('active');
        elements.startRecordBtn.disabled = true;
        elements.stopRecordBtn.disabled = false;
        showStatus('🎤 Recording... Speak now!', 'error');

        console.log('Recording started');
    } catch (error) {
        console.error('Error starting recording:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Stop recording
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());

        clearInterval(recordingInterval);
        elements.recordingIndicator.classList.remove('active');
        elements.startRecordBtn.disabled = false;
        elements.stopRecordBtn.disabled = true;

        showStatus('⏳ Processing audio through Ditto...', 'info');
    }
}

/**
 * Update recording time display
 */
function updateRecordingTime() {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    elements.recordingTime.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Setup audio visualizer
 */
function setupVisualizer(stream) {
    audioContext = new AudioContext();
    analyser = audioContext.createAnalyser();
    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);
    analyser.fftSize = 256;

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const canvas = elements.audioVisualizer;
    const ctx = canvas.getContext('2d');

    function draw() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            requestAnimationFrame(draw);
        }

        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;

            const gradient = ctx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height);
            gradient.addColorStop(0, '#667eea');
            gradient.addColorStop(1, '#764ba2');

            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);

            x += barWidth;
        }
    }

    draw();
}

/**
 * Process recorded audio through Ditto WebRTC pipeline
 */
async function processAudioThroughDitto() {
    try {
        processStartTime = Date.now();

        const serverUrl = elements.serverUrl.value;
        const avatarSource = elements.avatarSource.value;

        // Connect to WebSocket
        console.log('Connecting to server...');
        ws = new WebSocket(serverUrl);

        ws.onopen = async () => {
            console.log('WebSocket connected');

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
        };

    } catch (error) {
        console.error('Error processing audio:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Handle signaling messages
 */
async function handleSignalingMessage(message) {
    console.log('Received message:', message);

    switch (message.type) {
        case 'ready':
            showStatus('Server ready, creating connection...', 'info');
            await createPeerConnection();
            await createOffer();
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
    }
}

/**
 * Create WebRTC peer connection
 */
async function createPeerConnection() {
    const configuration = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
        ]
    };

    pc = new RTCPeerConnection(configuration);

    // Handle ICE candidates
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

    // Handle connection state
    pc.onconnectionstatechange = () => {
        console.log('Connection state:', pc.connectionState);

        if (pc.connectionState === 'connected') {
            showStatus('✓ Connected! Processing video...', 'success');
        } else if (pc.connectionState === 'failed') {
            showStatus('Connection failed', 'error');
        }
    };

    // Handle incoming media (video + audio)
    let remoteStream = null;
    let videoRecorder = null;
    let recordingChunks = [];
    let trackCount = 0;
    let recorderStarted = false; // Flag to prevent duplicate recorder creation

    pc.ontrack = (event) => {
        console.log('Received track:', event.track.kind);
        trackCount++;

        if (event.streams && event.streams[0]) {
            const stream = event.streams[0];

            // Initialize stream on first track
            if (!remoteStream) {
                remoteStream = stream;
            }

            // Wait for both audio and video tracks before starting recorder
            const audioTracks = remoteStream.getAudioTracks();
            const videoTracks = remoteStream.getVideoTracks();

            console.log(`Tracks received: audio=${audioTracks.length}, video=${videoTracks.length}`);

            // Start recorder only once when we have both tracks
            if (audioTracks.length > 0 && videoTracks.length > 0 && !recorderStarted) {
                recorderStarted = true; // Set flag immediately to prevent duplicate
                console.log('Both tracks ready, waiting for tracks to be active...');

                // Wait a moment for tracks to start producing frames
                setTimeout(() => {
                    console.log('Starting recorder...');

                    // Get fresh track references
                    const videoTrack = remoteStream.getVideoTracks()[0];
                    const audioTrack = remoteStream.getAudioTracks()[0];

                    if (!videoTrack || !audioTrack) {
                        console.error('Missing tracks!', { video: !!videoTrack, audio: !!audioTrack });
                        showStatus('Error: Missing tracks', 'error');
                        return;
                    }

                    console.log('Track states:', {
                        video: videoTrack.readyState,
                        audio: audioTrack.readyState
                    });

                    // Create a FRESH MediaStream with the current tracks
                    // This ensures MediaRecorder gets the right tracks
                    const recordStream = new MediaStream();
                    recordStream.addTrack(videoTrack);
                    recordStream.addTrack(audioTrack);
                    console.log('Created fresh MediaStream for recording');

                    // Try multiple codecs for compatibility
                    let mimeType = 'video/webm;codecs=vp8,opus';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = 'video/webm';
                        console.log('Falling back to default video/webm codec');
                    }

                    videoRecorder = new MediaRecorder(recordStream, { mimeType });
                    console.log('MediaRecorder created with mime:', mimeType);
                    console.log('MediaRecorder state:', videoRecorder.state);

                    videoRecorder.onstart = () => {
                        console.log('✓ MediaRecorder started event fired');
                    };

                    videoRecorder.onerror = (event) => {
                        console.error('MediaRecorder error:', event.error);
                        showStatus('MediaRecorder error: ' + event.error, 'error');
                    };

                    videoRecorder.ondataavailable = (e) => {
                        console.log(`ondataavailable fired: ${e.data.size} bytes`);
                        if (e.data.size > 0) {
                            recordingChunks.push(e.data);
                            console.log(`✓ Recorded chunk #${recordingChunks.length}: ${e.data.size} bytes`);
                        } else {
                            console.warn('Received empty chunk');
                        }
                    };

                    videoRecorder.onstop = () => {
                        console.log(`Recording stopped, total chunks: ${recordingChunks.length}`);

                        if (recordingChunks.length === 0) {
                            console.error('No video chunks recorded!');
                            showStatus('Error: No video data received', 'error');
                            return;
                        }

                        receivedVideoBlob = new Blob(recordingChunks, { type: mimeType });
                        console.log(`Video blob created: ${(receivedVideoBlob.size / 1024).toFixed(2)} KB, type: ${receivedVideoBlob.type}`);

                        if (receivedVideoBlob.size === 0) {
                            console.error('Video blob is empty!');
                            showStatus('Error: Video blob is empty', 'error');
                            return;
                        }

                        const videoUrl = URL.createObjectURL(receivedVideoBlob);

                        elements.resultVideo.src = videoUrl;
                        elements.resultVideo.muted = false; // Ensure audio is enabled
                        elements.resultVideo.volume = 1.0;
                        elements.playbackBtn.disabled = false;
                        elements.downloadBtn.disabled = false;

                        const processTime = ((Date.now() - processStartTime) / 1000).toFixed(1);
                        elements.statVideoSize.textContent = `${(receivedVideoBlob.size / 1024).toFixed(2)} KB`;
                        elements.statProcessTime.textContent = `${processTime}s`;

                        showStatus('✅ Video ready! Click Play Result to view', 'success');

                        // Wait a moment for video element to load
                        setTimeout(() => {
                            elements.resultVideo.play().catch(e => {
                                console.error('Autoplay failed:', e);
                                console.error('Video element state:', {
                                    src: elements.resultVideo.src,
                                    readyState: elements.resultVideo.readyState,
                                    networkState: elements.resultVideo.networkState,
                                    error: elements.resultVideo.error
                                });
                                showStatus('Video ready! Click the video to play', 'success');
                            });
                        }, 500);

                        console.log('Video ready:', videoUrl);
                    };

                    // Monitor track states
                    recordStream.getTracks().forEach((track, index) => {
                        console.log(`RecordStream Track ${index} (${track.kind}): enabled=${track.enabled}, readyState=${track.readyState}`);

                        track.onended = () => {
                            console.warn(`Track ${track.kind} ended!`);
                        };

                        track.onmute = () => {
                            console.warn(`Track ${track.kind} muted!`);
                        };

                        track.onunmute = () => {
                            console.log(`Track ${track.kind} unmuted`);
                        };
                    });

                    // Start recording
                    try {
                        videoRecorder.start(100); // Collect data every 100ms
                        console.log('MediaRecorder.start() called, state:', videoRecorder.state);
                    } catch (error) {
                        console.error('Failed to start MediaRecorder:', error);
                        showStatus('Failed to start recorder: ' + error.message, 'error');
                        return;
                    }

                    // Stop after audio duration + generous buffer for processing
                    const audioDuration = parseFloat(elements.statDuration.textContent) || 5;
                    const recordDuration = (audioDuration + 10) * 1000;

                    console.log(`Will record for ${(recordDuration/1000).toFixed(1)}s (audio=${audioDuration}s + buffer=10s)`);

                    const stopTimer = setTimeout(() => {
                        console.log('Stop timer fired, stopping recorder...');
                        console.log('Recorder state before stop:', videoRecorder ? videoRecorder.state : 'null');
                        if (videoRecorder && videoRecorder.state !== 'inactive') {
                            videoRecorder.stop();
                        }
                        cleanup();
                    }, recordDuration);

                    console.log(`Stop timer scheduled for ${recordDuration}ms from now`);

                }, 1000); // Wait 1 second for tracks to start producing frames
            }
        }
    };

    console.log('Peer connection created');
}

/**
 * Create and send offer with recorded audio
 */
async function createOffer() {
    try {
        // Create audio stream from recorded blob
        const audioTrack = await createAudioTrackFromBlob(recordedAudioBlob);
        pc.addTrack(audioTrack);

        // Add video transceiver to receive video
        pc.addTransceiver('video', { direction: 'recvonly' });

        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        ws.send(JSON.stringify({
            type: 'offer',
            sdp: pc.localDescription.sdp
        }));

        console.log('Sent offer with recorded audio');
    } catch (error) {
        console.error('Error creating offer:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Create MediaStreamTrack from recorded audio blob
 * IMPORTANT: We loop the audio to keep the track alive during video generation
 */
async function createAudioTrackFromBlob(blob) {
    // Convert blob to ArrayBuffer
    const arrayBuffer = await blob.arrayBuffer();

    // Decode audio
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Create MediaStreamDestination
    const destination = audioContext.createMediaStreamDestination();

    // Create multiple buffer sources to loop the audio
    let currentTime = audioContext.currentTime;
    const duration = audioBuffer.duration;

    function scheduleBuffer() {
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(destination);
        source.start(currentTime);

        currentTime += duration;

        // Schedule next iteration after a delay
        // This keeps the audio track alive even after recording finishes
        source.onended = () => {
            scheduleBuffer();
        };
    }

    scheduleBuffer();

    console.log(`Audio track created: ${duration.toFixed(2)}s, will loop to keep connection alive`);
    return destination.stream.getAudioTracks()[0];
}

/**
 * Handle SDP answer
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
        showStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Handle ICE candidate
 */
async function handleIceCandidate(candidate) {
    try {
        if (candidate) {
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
        }
    } catch (error) {
        console.error('Error adding ICE candidate:', error);
    }
}

/**
 * Cleanup connections
 */
function cleanup() {
    if (pc) {
        pc.close();
        pc = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
}

/**
 * Play the result video
 */
function playResult() {
    if (elements.resultVideo.src) {
        elements.resultVideo.play();
    }
}

/**
 * Download the result video
 */
function downloadResult() {
    if (receivedVideoBlob) {
        const url = URL.createObjectURL(receivedVideoBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ditto_recording_${Date.now()}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log('Video downloaded');
    }
}

/**
 * Show status message
 */
function showStatus(message, type) {
    elements.status.textContent = message;
    elements.status.className = `status-box ${type}`;
    elements.status.style.display = 'block';
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Event listeners
elements.startRecordBtn.addEventListener('click', startRecording);
elements.stopRecordBtn.addEventListener('click', stopRecording);
elements.playbackBtn.addEventListener('click', playResult);
elements.downloadBtn.addEventListener('click', downloadResult);
