// WebSocket Message Types
export type WSMessageType =
  | 'init'
  | 'audio_chunk'
  | 'stop'
  | 'status'
  | 'frame'
  | 'progress'
  | 'error'
  | 'complete'
  | 'heartbeat';

export interface WSMessage {
  type: WSMessageType;
}

export interface WSInitMessage extends WSMessage {
  type: 'init';
  source: string; // base64 encoded image/video
  audio?: string; // base64 encoded audio file (optional)
  config?: StreamConfig;
}

export interface WSAudioChunkMessage extends WSMessage {
  type: 'audio_chunk';
  data: string; // base64 encoded PCM audio
  timestamp: number;
}

export interface WSStopMessage extends WSMessage {
  type: 'stop';
}

export interface WSStatusMessage extends WSMessage {
  type: 'status';
  message: string;
  session_id?: string;
}

export interface WSFrameMessage extends WSMessage {
  type: 'frame';
  data: string; // base64 encoded JPEG
  frame_id: number;
  timestamp: number;
}

export interface WSProgressMessage extends WSMessage {
  type: 'progress';
  processed: number;
  fps: number;
  latency_ms: number;
}

export interface WSErrorMessage extends WSMessage {
  type: 'error';
  message: string;
  code?: string;
}

export interface WSCompleteMessage extends WSMessage {
  type: 'complete';
  total_frames: number;
  duration_seconds: number;
}

// Configuration Types
export interface StreamConfig {
  max_size?: number;
  crop_scale?: number;
  crop_vx_ratio?: number;
  crop_vy_ratio?: number;
  emo?: number;
  sampling_timesteps?: number;
  fps?: number;
}

// App State Types
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

export type SessionStatus = 'idle' | 'initializing' | 'ready' | 'streaming' | 'completed' | 'error';

export interface AppState {
  connectionStatus: ConnectionStatus;
  sessionStatus: SessionStatus;
  sessionId?: string;
  framesProcessed: number;
  fps: number;
  latency: number;
  error?: string;
}

// API Types
export interface HealthResponse {
  status: string;
  version: string;
  backend: string;
  gpu_available: boolean;
  active_sessions: number;
  uptime_seconds: number;
}

export interface ModelInfo {
  name: string;
  backend: string;
  loaded: boolean;
  device: string;
}
