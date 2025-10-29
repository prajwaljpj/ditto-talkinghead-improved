import { Settings } from 'lucide-react';
import type { StreamConfig } from '../types';

interface ConfigPanelProps {
  config: StreamConfig;
  onChange: (config: StreamConfig) => void;
  disabled?: boolean;
}

export function ConfigPanel({ config, onChange, disabled }: ConfigPanelProps) {
  const updateConfig = (key: keyof StreamConfig, value: number) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Settings className="w-5 h-5 text-gray-700" />
        <h3 className="text-lg font-semibold text-gray-800">Configuration</h3>
      </div>

      <div className="space-y-4">
        {/* Emotion */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Emotion (0-7)
            <span className="ml-2 text-gray-500 font-normal">Current: {config.emo || 4}</span>
          </label>
          <input
            type="range"
            min="0"
            max="7"
            value={config.emo || 4}
            onChange={(e) => updateConfig('emo', parseInt(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Neutral</span>
            <span>Expressive</span>
          </div>
        </div>

        {/* Crop Scale */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Crop Scale
            <span className="ml-2 text-gray-500 font-normal">Current: {config.crop_scale || 2.3}</span>
          </label>
          <input
            type="range"
            min="1.5"
            max="3.0"
            step="0.1"
            value={config.crop_scale || 2.3}
            onChange={(e) => updateConfig('crop_scale', parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1.5</span>
            <span>3.0</span>
          </div>
        </div>

        {/* Sampling Timesteps */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quality (Sampling Steps)
            <span className="ml-2 text-gray-500 font-normal">Current: {config.sampling_timesteps || 50}</span>
          </label>
          <input
            type="range"
            min="20"
            max="100"
            step="10"
            value={config.sampling_timesteps || 50}
            onChange={(e) => updateConfig('sampling_timesteps', parseInt(e.target.value))}
            disabled={disabled}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600 disabled:opacity-50"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Fast</span>
            <span>High Quality</span>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Higher values improve quality but increase processing time
          </p>
        </div>

        {/* Max Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Max Resolution
            <span className="ml-2 text-gray-500 font-normal">Current: {config.max_size || 1920}px</span>
          </label>
          <select
            value={config.max_size || 1920}
            onChange={(e) => updateConfig('max_size', parseInt(e.target.value))}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="512">512px (Fast)</option>
            <option value="1024">1024px (Balanced)</option>
            <option value="1920">1920px (High Quality)</option>
          </select>
        </div>

        {/* FPS */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Output FPS
            <span className="ml-2 text-gray-500 font-normal">Current: {config.fps || 25}</span>
          </label>
          <select
            value={config.fps || 25}
            onChange={(e) => updateConfig('fps', parseInt(e.target.value))}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="15">15 FPS</option>
            <option value="24">24 FPS (Cinema)</option>
            <option value="25">25 FPS (PAL)</option>
            <option value="30">30 FPS (NTSC)</option>
          </select>
        </div>
      </div>
    </div>
  );
}
