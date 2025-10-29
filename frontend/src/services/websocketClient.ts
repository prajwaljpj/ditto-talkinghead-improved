import type {
  WSMessage,
  WSInitMessage,
  WSAudioChunkMessage,
  WSStopMessage,
  WSStatusMessage,
  StreamConfig,
} from '../types';

type MessageHandler = (message: WSMessage) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Map<string, MessageHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private reconnectDelay = 2000;

  constructor(url: string) {
    this.url = url;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WSMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket closed');
          this.handleClose();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private handleClose() {
    this.emit('status', {
      type: 'status',
      message: 'Connection closed',
    } as WSStatusMessage);

    // Attempt reconnection if within limits
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
        this.connect().catch(console.error);
      }, this.reconnectDelay);
    }
  }

  private handleMessage(message: WSMessage) {
    const handlers = this.handlers.get(message.type) || [];
    handlers.forEach((handler) => handler(message));

    // Also emit to 'all' handlers
    const allHandlers = this.handlers.get('all') || [];
    allHandlers.forEach((handler) => handler(message));
  }

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, []);
    }
    this.handlers.get(type)!.push(handler);
  }

  off(type: string, handler: MessageHandler) {
    const handlers = this.handlers.get(type);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(type: string, message: WSMessage) {
    const handlers = this.handlers.get(type) || [];
    handlers.forEach((handler) => handler(message));
  }

  send(message: WSMessage) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.error('WebSocket is not connected');
      throw new Error('WebSocket is not connected');
    }
  }

  sendInit(source: string, config?: StreamConfig, audio?: string) {
    const message: WSInitMessage = {
      type: 'init',
      source,
      audio,
      config,
    };
    this.send(message);
  }

  sendAudioChunk(data: string, timestamp: number) {
    const message: WSAudioChunkMessage = {
      type: 'audio_chunk',
      data,
      timestamp,
    };
    this.send(message);
  }

  sendStop() {
    const message: WSStopMessage = {
      type: 'stop',
    };
    this.send(message);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.handlers.clear();
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let client: WebSocketClient | null = null;

export function getWebSocketClient(): WebSocketClient {
  if (!client) {
    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.DEV ? '8000' : window.location.port;
    const url = `${protocol}//${host}:${port}/api/v1/stream/ws`;
    client = new WebSocketClient(url);
  }
  return client;
}
