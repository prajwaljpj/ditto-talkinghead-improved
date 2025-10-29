import { Activity, Wifi, WifiOff, AlertCircle } from 'lucide-react';
import type { AppState } from '../types';

interface StatusBarProps {
  state: AppState;
}

export function StatusBar({ state }: StatusBarProps) {
  const getConnectionIcon = () => {
    switch (state.connectionStatus) {
      case 'connected':
        return <Wifi className="w-4 h-4 text-green-500" />;
      case 'connecting':
        return <Activity className="w-4 h-4 text-yellow-500 animate-pulse" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <WifiOff className="w-4 h-4 text-gray-400" />;
    }
  };

  const getConnectionText = () => {
    switch (state.connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'error':
        return 'Connection Error';
      default:
        return 'Disconnected';
    }
  };

  const getSessionStatusColor = () => {
    switch (state.sessionStatus) {
      case 'streaming':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'initializing':
      case 'ready':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'completed':
        return 'bg-purple-100 text-purple-800 border-purple-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Connection Status */}
        <div className="flex items-center space-x-3">
          {getConnectionIcon()}
          <div>
            <p className="text-xs text-gray-500">Connection</p>
            <p className="text-sm font-medium text-gray-800">{getConnectionText()}</p>
          </div>
        </div>

        {/* Session Status */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Session Status</p>
          <span className={`inline-block px-3 py-1 text-xs font-medium rounded-full border ${getSessionStatusColor()}`}>
            {state.sessionStatus.charAt(0).toUpperCase() + state.sessionStatus.slice(1)}
          </span>
        </div>

        {/* Frames Processed */}
        <div>
          <p className="text-xs text-gray-500">Frames Processed</p>
          <p className="text-lg font-semibold text-gray-800">{state.framesProcessed}</p>
        </div>

        {/* Performance Stats */}
        <div>
          <p className="text-xs text-gray-500">Performance</p>
          <div className="flex items-baseline space-x-2">
            <span className="text-lg font-semibold text-gray-800">{state.fps.toFixed(1)}</span>
            <span className="text-xs text-gray-500">FPS</span>
            <span className="text-gray-300">|</span>
            <span className="text-sm text-gray-600">{state.latency.toFixed(0)}ms</span>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {state.error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{state.error}</p>
        </div>
      )}

      {/* Session ID */}
      {state.sessionId && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Session ID: <span className="font-mono text-gray-700">{state.sessionId}</span>
          </p>
        </div>
      )}
    </div>
  );
}
