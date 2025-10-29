import { useRef, useEffect } from 'react';
import { Download } from 'lucide-react';

interface VideoDisplayProps {
  currentFrame?: string;
  isProcessing: boolean;
  outputPath?: string;
  sessionId?: string;
}

export function VideoDisplay({ currentFrame, isProcessing, outputPath, sessionId }: VideoDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (currentFrame && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const img = new Image();
      img.onload = () => {
        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
      };
      img.src = currentFrame;
    }
  }, [currentFrame]);

  const handleDownload = () => {
    if (sessionId) {
      const apiUrl = import.meta.env.DEV
        ? 'http://localhost:8000'
        : window.location.origin;
      window.open(`${apiUrl}/api/v1/outputs/${sessionId}_output.mp4`, '_blank');
    }
  };

  return (
    <div className="space-y-4">
      {/* Canvas Display */}
      <div className="relative bg-gray-900 rounded-lg overflow-hidden min-h-[400px] flex items-center justify-center">
        {currentFrame ? (
          <canvas
            ref={canvasRef}
            className="max-w-full max-h-[600px] h-auto"
          />
        ) : (
          <div className="text-gray-400 text-center p-8">
            <p className="text-lg mb-2">
              {isProcessing ? 'Generating video...' : 'No video to display'}
            </p>
            {isProcessing && (
              <div className="mt-4">
                <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-500 border-r-transparent"></div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Download Button */}
      {outputPath && !isProcessing && (
        <div className="flex justify-center">
          <button
            onClick={handleDownload}
            className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium shadow-lg"
          >
            <Download className="w-5 h-5" />
            <span>Download Video</span>
          </button>
        </div>
      )}

      {/* Processing Info */}
      {isProcessing && currentFrame && (
        <div className="text-center text-sm text-gray-500">
          <p>Processing... Frames will appear here in real-time</p>
        </div>
      )}
    </div>
  );
}
