# 🚀 Ditto Talking Head - Quick Start Guide

Get up and running in 5 minutes!

## ✅ Prerequisites

- Python 3.10
- Node.js 16+ (for frontend)
- CUDA-capable GPU (recommended)
- ~20GB disk space

## 🎯 Quick Start (3 Steps)

### Step 1: Start the Backend API

```bash
# Install Python dependencies (already done if you've run uv sync)
# uv sync

# Start the API server
python start_api.py
```

✅ **API will be running at http://localhost:8000**

### Step 2: Start the Frontend

```bash
# Open a new terminal
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

✅ **Frontend will be running at http://localhost:3000**

### Step 3: Use the Application

1. **Open your browser**: http://localhost:3000
2. **Upload an image**: Drag & drop or click to browse
3. **Configure settings**: Adjust emotion, quality, resolution
4. **Click "Start Streaming"**: Initialize WebSocket connection
5. **Monitor status**: Watch real-time updates

---

## 📊 Verify Everything Works

### Check API Health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "backend": "pytorch",
  "gpu_available": true,
  "active_sessions": 0
}
```

### Run API Tests

```bash
python test_api.py
```

Expected: **6/6 tests passed** ✅

### Check Frontend Build

```bash
cd frontend
npm run build
```

Expected: **Build successful** ✅

---

## 🎨 What You'll See

### Main Interface

```
┌──────────────────────────────────────────────┐
│  Ditto Talking Head                          │
│  Real-time Talking Head Synthesis            │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│  Status Bar                                   │
│  ● Connected | Ready | 0 frames | 0 FPS      │
└──────────────────────────────────────────────┘

┌────────────────┐  ┌───────────────────────┐
│  Source Image  │  │  Output Display       │
│                │  │                       │
│  [Drop Zone]   │  │  [Generated Video]    │
│                │  │                       │
│  Configuration │  │  [Status Info]        │
│  - Emotion: 4  │  │                       │
│  - Quality: 50 │  │                       │
│  - FPS: 25     │  │                       │
│                │  │                       │
│  [Start]       │  │                       │
└────────────────┘  └───────────────────────┘
```

---

## 🛠️ Troubleshooting

### API Won't Start

**Problem**: Port 8000 already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use a different port
python start_api.py --port 8080
```

**Problem**: Models not found

```bash
# Check if models exist
ls checkpoints/ditto_cfg/

# If missing, download them (refer to main README)
```

### Frontend Won't Start

**Problem**: Port 3000 already in use

```bash
# Vite will automatically try the next available port
# Or edit vite.config.ts to change the default port
```

**Problem**: Build errors

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### WebSocket Connection Fails

**Problem**: CORS errors

```bash
# Check .env file
# Make sure DITTO_CORS_ORIGINS includes http://localhost:3000
```

**Problem**: Connection refused

```bash
# Ensure API is running
curl http://localhost:8000/api/v1/health

# Check logs
tail -f logs/latest.json
```

---

## 📖 Next Steps

### Test the Demo
1. Upload a portrait image
2. Adjust settings
3. Click "Start Streaming"
4. Observe WebSocket connection
5. Monitor status updates

### Add Audio (Future Enhancement)
- Implement microphone capture
- Add audio file upload
- Stream audio chunks to API
- Display generated frames

### Deploy to Production
- Build frontend: `cd frontend && npm run build`
- Serve with nginx or similar
- Configure environment variables
- Set up SSL/TLS

---

## 🎯 Common Use Cases

### Local Development
```bash
# Terminal 1: API with auto-reload
python start_api.py --reload

# Terminal 2: Frontend with hot-reload
cd frontend && npm run dev
```

### Testing
```bash
# Test API
python test_api.py

# Test frontend build
cd frontend && npm run build

# Preview production build
npm run preview
```

### Production
```bash
# Build frontend
cd frontend && npm run build

# Serve static files (frontend/dist/)
# Run API in production mode
python start_api.py --workers 1

# Use process manager (systemd, supervisor, etc.)
```

---

## 🔍 Useful Commands

### API
```bash
# Start with custom settings
python start_api.py --host 0.0.0.0 --port 8080 --reload

# View API logs
tail -f logs/latest.json | jq .

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Frontend
```bash
# Development with hot-reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

### Health Checks
```bash
# API health
curl http://localhost:8000/api/v1/health

# GPU info
curl http://localhost:8000/api/v1/gpu-info

# Models list
curl http://localhost:8000/api/v1/models

# Session stats
curl http://localhost:8000/api/v1/stream/stats
```

---

## 📚 Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **API README**: `API_README.md`
- **Frontend README**: `frontend/README.md`
- **Implementation Details**: `FINAL_IMPLEMENTATION_SUMMARY.md`

---

## 💡 Tips

1. **Use Chrome DevTools** to inspect WebSocket messages
2. **Check browser console** for frontend errors
3. **Monitor API logs** for backend issues
4. **Test with small images first** (faster processing)
5. **Adjust quality settings** based on your GPU

---

## ✅ Success Checklist

- [ ] Backend API running (http://localhost:8000)
- [ ] Frontend running (http://localhost:3000)
- [ ] API health check passes
- [ ] API tests pass (6/6)
- [ ] Frontend builds successfully
- [ ] WebSocket connection works
- [ ] Image upload works
- [ ] Configuration changes work
- [ ] Status updates work

---

## 🎉 You're All Set!

Your Ditto Talking Head system is now running!

**What's Working:**
- ✅ Backend API with WebSocket streaming
- ✅ Frontend UI with real-time updates
- ✅ Session management
- ✅ Health monitoring
- ✅ Comprehensive logging

**Ready For:**
- 🔄 Testing with real workloads
- 🎤 Adding audio streaming
- 📹 Adding frame display
- 🐳 Docker deployment

Enjoy your talking head synthesis system! 🚀
