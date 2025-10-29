# Ditto Talking Head - Frontend

Modern React frontend for the Ditto Talking Head API, built with TypeScript, Vite, and TailwindCSS.

## Features

- 🎨 **Modern UI** - Clean, responsive interface with TailwindCSS
- 🔌 **WebSocket Integration** - Real-time streaming communication
- 📤 **Drag & Drop Upload** - Easy source image upload
- ⚙️ **Configuration Panel** - Adjust inference settings
- 📊 **Live Status** - Real-time progress and performance monitoring
- 🎯 **Type-Safe** - Full TypeScript support

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool & dev server
- **TailwindCSS** - Utility-first CSS
- **Lucide React** - Icon library

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at http://localhost:3000

### 3. Ensure API is Running

Make sure the backend API is running on http://localhost:8000:

```bash
cd ..
python start_api.py
```

## Project Structure

```
frontend/
├── src/
│   ├── components/      # React components
│   │   ├── FileUpload.tsx      # Image upload with drag & drop
│   │   ├── ConfigPanel.tsx     # Configuration settings
│   │   └── StatusBar.tsx       # Status display
│   ├── services/        # Service layer
│   │   └── websocketClient.ts  # WebSocket client
│   ├── types/           # TypeScript types
│   │   └── index.ts            # Type definitions
│   ├── utils/           # Utility functions
│   │   └── imageUtils.ts       # Image processing
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript config
├── vite.config.ts       # Vite configuration
└── tailwind.config.js   # TailwindCSS config
```

## Components

### FileUpload

Handles source image upload with:
- Drag & drop support
- Image validation (type, size)
- Auto-resize to max dimensions
- Base64 conversion
- Preview display

### ConfigPanel

Configuration interface for:
- Emotion selection (0-7)
- Crop scale adjustment
- Quality settings (sampling timesteps)
- Resolution selection
- FPS configuration

### StatusBar

Real-time monitoring of:
- Connection status
- Session status
- Frames processed
- Performance metrics (FPS, latency)
- Error messages

## WebSocket Protocol

The frontend communicates with the backend via WebSocket using this protocol:

### Client → Server

```typescript
// Initialize session
{
  type: 'init',
  source: '<base64-image>',
  config: { emo: 4, crop_scale: 2.3, ... }
}

// Send audio chunk
{
  type: 'audio_chunk',
  data: '<base64-audio>',
  timestamp: 123.45
}

// Stop streaming
{
  type: 'stop'
}
```

### Server → Client

```typescript
// Status update
{
  type: 'status',
  message: 'Session initialized',
  session_id: 'abc-123'
}

// Progress update
{
  type: 'progress',
  processed: 100,
  fps: 25.5,
  latency_ms: 45.2
}

// Generated frame
{
  type: 'frame',
  data: '<base64-jpeg>',
  frame_id: 42,
  timestamp: 1.68
}

// Error
{
  type: 'error',
  message: 'Error description',
  code: 'ERROR_CODE'
}

// Complete
{
  type: 'complete',
  total_frames: 500,
  duration_seconds: 20.5
}
```

## Available Scripts

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## Development

### Adding New Components

1. Create component in `src/components/`
2. Export from component file
3. Import in `App.tsx`

### Adding New Types

1. Add types to `src/types/index.ts`
2. Export from the file
3. Import where needed

### Styling

Use TailwindCSS utility classes:

```tsx
<div className="bg-white rounded-lg shadow-sm p-4">
  <h2 className="text-lg font-semibold text-gray-800">Title</h2>
</div>
```

Custom colors are defined in `tailwind.config.js`:
- `primary-*` - Blue shades for primary UI elements

## Configuration

### API Endpoint

The WebSocket URL is automatically determined from the current location:

- **Development**: `ws://localhost:8000/api/v1/stream/ws`
- **Production**: Based on deployment URL

To change the API URL, modify `src/services/websocketClient.ts`:

```typescript
const port = import.meta.env.DEV ? '8000' : window.location.port;
```

### Vite Proxy

API requests are proxied through Vite in development (see `vite.config.ts`):

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## Building for Production

```bash
# Build optimized bundle
npm run build

# Output will be in dist/
# Serve with any static file server
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

Requires:
- WebSocket support
- ES2020 features
- FileReader API
- Canvas API

## Known Limitations

1. **Audio Streaming** - Not yet implemented. The current version demonstrates the connection flow but doesn't stream audio to the API.

2. **Frame Display** - Generated frames are not yet displayed. You'll need to add:
   - Canvas or video element for frame rendering
   - Frame buffer management
   - Playback controls

3. **Microphone Input** - No microphone capture yet. To add:
   - Use `navigator.mediaDevices.getUserMedia()`
   - Capture audio chunks
   - Convert to PCM and base64
   - Send via WebSocket

## Future Enhancements

- [ ] Add microphone input
- [ ] Implement frame display canvas
- [ ] Add video recording/download
- [ ] Add audio file upload
- [ ] Implement playback controls
- [ ] Add session history
- [ ] Implement frame buffering
- [ ] Add quality presets
- [ ] Mobile responsive improvements

## Troubleshooting

### WebSocket Connection Fails

1. Ensure API is running: `python start_api.py`
2. Check API health: http://localhost:8000/api/v1/health
3. Check CORS settings in API config

### Image Upload Fails

1. Check file size (<50MB)
2. Ensure valid format (JPEG, PNG, WebP)
3. Check browser console for errors

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

## License

See main repository LICENSE file.
