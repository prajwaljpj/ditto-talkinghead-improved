import { X, Music } from 'lucide-react';
import { useState, useRef } from 'react';

interface AudioUploadProps {
  onFileSelect: (audioBase64: string, duration: number) => void;
  disabled?: boolean;
}

export function AudioUpload({ onFileSelect, disabled }: AudioUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [error, setError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAudioFile = (file: File): string | null => {
    // Check file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm'];
    if (!validTypes.includes(file.type)) {
      return 'Please upload a valid audio file (WAV, MP3, OGG, or WebM)';
    }

    // Check file size (max 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      return 'Audio file is too large (max 50MB)';
    }

    return null;
  };

  const getAudioDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const audio = new Audio();
      const url = URL.createObjectURL(file);

      audio.addEventListener('loadedmetadata', () => {
        URL.revokeObjectURL(url);
        resolve(audio.duration);
      });

      audio.addEventListener('error', () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load audio'));
      });

      audio.src = url;
    });
  };

  const processFile = async (file: File) => {
    const validationError = validateAudioFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError('');
    setAudioFile(file);

    try {
      // Get audio duration
      const audioDuration = await getAudioDuration(file);
      setDuration(audioDuration);

      // Convert to base64
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target?.result as string;
        onFileSelect(base64, audioDuration);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setError('Failed to process audio file');
      console.error(err);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  const handleRemove = () => {
    setAudioFile(null);
    setDuration(0);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Audio File
        </label>

        {!audioFile ? (
          <div
            className={`
              border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
              transition-colors duration-200
              ${isDragging
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400 bg-gray-50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            onDragOver={(e) => {
              e.preventDefault();
              if (!disabled) setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => !disabled && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={handleFileInput}
              disabled={disabled}
            />

            <Music className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p className="text-gray-600 mb-2">
              Drag & drop your audio file here
            </p>
            <p className="text-sm text-gray-500">
              or click to browse
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Supported: WAV, MP3, OGG, WebM (max 50MB)
            </p>
          </div>
        ) : (
          <div className="border-2 border-green-300 bg-green-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0">
                  <Music className="w-8 h-8 text-green-600" />
                </div>
                <div>
                  <p className="font-medium text-gray-800">{audioFile.name}</p>
                  <p className="text-sm text-gray-600">
                    {(audioFile.size / 1024 / 1024).toFixed(2)} MB · {formatDuration(duration)}
                  </p>
                </div>
              </div>
              <button
                onClick={handleRemove}
                disabled={disabled}
                className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-100 rounded-full transition-colors disabled:opacity-50"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}
    </div>
  );
}
