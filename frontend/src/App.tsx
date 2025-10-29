import { useState, useEffect, useRef } from 'react';
import { Play, Square, Loader2 } from 'lucide-react';
import { FileUpload } from './components/FileUpload';
import { AudioUpload } from './components/AudioUpload';
import { ConfigPanel } from './components/ConfigPanel';
import { StatusBar } from './components/StatusBar';
import { VideoDisplay } from './components/VideoDisplay';
import { getWebSocketClient } from './services/websocketClient';
import type {
  AppState,
  StreamConfig,
  WSStatusMessage,
  WSProgressMessage,
  WSFrameMessage,
  WSErrorMessage,
  WSCompleteMessage,
} from './types';

const DEFAULT_CONFIG: StreamConfig = {
  emo: 4,
  crop_scale: 2.3,
  sampling_timesteps: 50,
  max_size: 1920,
  fps: 25,
};

function App() {
  const [sourceBase64, setSourceBase64] = useState<string>('');
  const [sourcePreview, setSourcePreview] = useState<string>('');
  const [audioBase64, setAudioBase64] = useState<string>('');
  const [_audioDuration, setAudioDuration] = useState<number>(0);
  const [config, setConfig] = useState<StreamConfig>(DEFAULT_CONFIG);
  const [currentFrame, setCurrentFrame] = useState<string>('');
  const [outputPath, setOutputPath] = useState<string>('');
  const [state, setState] = useState<AppState>({
    connectionStatus: 'disconnected',
    sessionStatus: 'idle',
    framesProcessed: 0,
    fps: 0,
    latency: 0,
  });

  const wsClient = useRef(getWebSocketClient());

  useEffect(() => {
    const client = wsClient.current;

    // Setup WebSocket event handlers
    client.on('status', (message) => {
      const msg = message as WSStatusMessage;
      console.log('Status:', msg.message);
      if (msg.session_id) {
        setState((prev) => ({ ...prev, sessionId: msg.session_id }));
      }
    });

    client.on('progress', (message) => {
      const msg = message as WSProgressMessage;
      setState((prev) => ({
        ...prev,
        framesProcessed: msg.processed,
        fps: msg.fps,
        latency: msg.latency_ms,
      }));
    });

    client.on('frame', (message) => {
      const msg = message as WSFrameMessage;
      setCurrentFrame(msg.data);
    });

    client.on('error', (message) => {
      const msg = message as WSErrorMessage;
      console.error('Error:', msg.message);
      setState((prev) => ({
        ...prev,
        sessionStatus: 'error',
        error: msg.message,
      }));
    });

    client.on('complete', (message) => {
      const msg = message as WSCompleteMessage;
      console.log('Complete:', msg);
      if ((msg as any).output_path) {
        setOutputPath((msg as any).output_path);
      }
      setState((prev) => ({
        ...prev,
        sessionStatus: 'completed',
      }));
    });

    return () => {
      client.disconnect();
    };
  }, []);

  const handleConnect = async () => {
    if (!sourceBase64) {
      setState((prev) => ({
        ...prev,
        error: 'Please upload a source image first',
      }));
      return;
    }

    try {
      setState((prev) => ({
        ...prev,
        connectionStatus: 'connecting',
        error: undefined,
      }));

      const client = wsClient.current;
      await client.connect();

      setState((prev) => ({
        ...prev,
        connectionStatus: 'connected',
        sessionStatus: 'initializing',
      }));

      // Send init message with audio
      client.sendInit(sourceBase64, config, audioBase64);

      setState((prev) => ({
        ...prev,
        sessionStatus: 'ready',
      }));
    } catch (error) {
      console.error('Connection error:', error);
      setState((prev) => ({
        ...prev,
        connectionStatus: 'error',
        error: error instanceof Error ? error.message : 'Connection failed',
      }));
    }
  };

  const handleDisconnect = () => {
    const client = wsClient.current;
    client.sendStop();
    client.disconnect();

    setState({
      connectionStatus: 'disconnected',
      sessionStatus: 'idle',
      framesProcessed: 0,
      fps: 0,
      latency: 0,
    });
  };

  const isConnected = state.connectionStatus === 'connected';
  const canStart = sourceBase64 && audioBase64 && !isConnected;
  const isProcessing = state.sessionStatus === 'initializing' || state.sessionStatus === 'streaming';

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Ditto Talking Head</h1>
              <p className="text-sm text-gray-500">Real-time Talking Head Synthesis</p>
            </div>
            <div className="flex items-center space-x-4">
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-primary-600 hover:text-primary-700"
              >
                API Docs
              </a>
              <div className="px-3 py-1 bg-primary-100 text-primary-700 text-xs font-medium rounded-full">
                v1.0.0
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Status Bar */}
        <div className="mb-6">
          <StatusBar state={state} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Upload & Config */}
          <div className="lg:col-span-1 space-y-6">
            {/* File Upload */}
            <div>
              <h2 className="text-lg font-semibold text-gray-800 mb-3">Source Image</h2>
              <FileUpload
                onFileSelect={(base64, preview) => {
                  setSourceBase64(base64);
                  setSourcePreview(preview);
                  setState((prev) => ({ ...prev, error: undefined }));
                }}
                disabled={isConnected}
              />
            </div>

            {/* Audio Upload */}
            <div>
              <h2 className="text-lg font-semibold text-gray-800 mb-3">Audio File</h2>
              <AudioUpload
                onFileSelect={(base64, duration) => {
                  setAudioBase64(base64);
                  setAudioDuration(duration);
                  setState((prev) => ({ ...prev, error: undefined }));
                }}
                disabled={isConnected}
              />
            </div>

            {/* Configuration */}
            <ConfigPanel
              config={config}
              onChange={setConfig}
              disabled={isConnected}
            />

            {/* Control Buttons */}
            <div className="space-y-3">
              <button
                onClick={handleConnect}
                disabled={!canStart || isProcessing}
                className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    <span>Start Streaming</span>
                  </>
                )}
              </button>

              {isConnected && (
                <button
                  onClick={handleDisconnect}
                  className="w-full flex items-center justify-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
                >
                  <Square className="w-5 h-5" />
                  <span>Stop & Disconnect</span>
                </button>
              )}
            </div>
          </div>

          {/* Right Column - Preview/Output */}
          <div className="lg:col-span-2">
            <h2 className="text-lg font-semibold text-gray-800 mb-3">Output</h2>
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              {!sourcePreview ? (
                <div className="flex items-center justify-center min-h-[600px]">
                  <div className="text-center text-gray-400">
                    <p className="text-lg">No source image uploaded</p>
                    <p className="text-sm mt-2">Upload an image and audio file to get started</p>
                  </div>
                </div>
              ) : state.sessionStatus === 'idle' ? (
                <div className="flex flex-col items-center justify-center min-h-[600px] space-y-4">
                  <img
                    src={sourcePreview}
                    alt="Source preview"
                    className="max-w-full max-h-96 rounded-lg shadow-md"
                  />
                  <p className="text-gray-600">Upload audio and click "Start Streaming" to begin</p>
                </div>
              ) : (
                <VideoDisplay
                  currentFrame={currentFrame}
                  isProcessing={state.sessionStatus === 'streaming' || state.sessionStatus === 'initializing'}
                  outputPath={outputPath}
                  sessionId={state.sessionId}
                />
              )}
            </div>
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 p-6 bg-white rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">How to Use</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-600">
            <li>Upload a source image of a face (JPEG, PNG, or WebP)</li>
            <li>Adjust configuration settings (emotion, quality, resolution)</li>
            <li>Click "Start Streaming" to initialize the session</li>
            <li>The system will connect to the backend API via WebSocket</li>
            <li>Generated frames will be displayed in real-time (when audio is streamed)</li>
            <li>Click "Stop & Disconnect" to end the session</li>
          </ol>
          <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              <strong>Demo Mode:</strong> This interface demonstrates the WebSocket connection and configuration.
              To complete the implementation, you'll need to add audio streaming functionality (microphone input or file upload)
              and frame display logic.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-500">
          <p>Ditto Talking Head API • Built with FastAPI & React</p>
          <p className="mt-1">Using Motion-Space Diffusion for Real-time Synthesis</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
