# Warning Logs Explanation

## Overview

This document explains the warning and debug logs you may see during normal operation of the LiveKit Gemini Agent.

## Common Warning Patterns

### 1. "Gemini receive loop ended - reconnecting..."

**What it means:**
- The Gemini Live API WebSocket connection closed normally
- The receive loop that listens for Gemini responses has ended
- The agent is automatically reconnecting to maintain the session

**Why it happens:**
- Normal WebSocket lifecycle - connections can close for various reasons:
  - Network hiccups
  - Server-side connection management
  - Timeouts
  - Graceful shutdowns

**Is it a problem?**
- **No** - This is expected behavior
- The agent has automatic reconnection logic
- The session will reconnect within 2 seconds
- No user action needed

**Log level:** WARNING (but it's informational, not an error)

---

### 2. "Failed to send audio to Gemini: sent 1000 (OK); then received 1000 (OK)"

**What it means:**
- The agent tried to send audio to Gemini
- But the WebSocket connection was already closed (code 1000 = normal closure)
- This is a **race condition** during reconnection

**Why it happens:**
1. Gemini session closes (normal WebSocket close)
2. Receive loop detects closure and starts reconnecting
3. **Meanwhile**, user audio is still being processed
4. Code tries to send audio to the closed session
5. WebSocket library raises an exception

**Timeline:**
```
Time 0ms:  User speaks → audio processing starts
Time 5ms:  Gemini session closes (receive loop ends)
Time 6ms:  Reconnection logic starts
Time 10ms: Audio processing finishes → tries to send
Time 11ms: ERROR: Connection already closed!
```

**Is it a problem?**
- **No** - This is a harmless race condition
- The audio chunk is simply dropped (will be sent on next reconnect)
- Reconnection happens automatically
- User won't notice (Gemini handles buffering)

**Fix applied:**
- Now handles connection closed errors gracefully
- Logs at DEBUG level instead of WARNING
- Skips the audio chunk silently during reconnection

---

## Debug Logs Explained

### WebSocket Communication Logs

```
DEBUG:websockets.client:> TEXT '{"realtime_input": {"media_chunks": [...]}}'
```
- **Meaning:** Sending audio data to Gemini
- **Frequency:** Every audio frame (typically 20ms chunks)
- **Action:** None needed - this is normal operation

```
DEBUG:websockets.client:< CLOSE 1000 (OK)
```
- **Meaning:** WebSocket connection closing normally
- **Code 1000:** Normal closure (not an error)
- **Action:** None needed - reconnection will happen automatically

```
DEBUG:websockets.client:= connection is CLOSED
```
- **Meaning:** WebSocket connection is now closed
- **Action:** None needed - reconnection logic will handle it

---

## Log Levels

### DEBUG Level
- **WebSocket communication:** All WebSocket messages
- **Connection state changes:** Opening, closing, reconnecting
- **Profiling information:** Stage-by-stage timing
- **Normal operation:** These are informational, not errors

### INFO Level
- **State transitions:** IDLE ↔ SPEAKING
- **Session lifecycle:** Starting, stopping, reconnecting
- **Periodic summaries:** Frame counts, chunk counts

### WARNING Level
- **Reconnection events:** "Gemini receive loop ended - reconnecting..."
  - This is actually normal behavior, just logged as warning for visibility
- **Connection issues:** Temporary connection problems (auto-recovered)

### ERROR Level
- **Authentication failures:** Permission errors, credential issues
- **Fatal errors:** Errors that prevent the agent from functioning

---

## Normal Operation Flow

### Healthy Session
```
✅ Gemini Live session started
📡 Listening for Gemini responses...
📨 Received response: <class 'google.genai.types.LiveServerMessage'>
📨 Received response: <class 'google.genai.types.LiveServerMessage'>
... (continuous responses)
```

### Reconnection Flow (Normal)
```
⚠️  Gemini receive loop ended - reconnecting...
🔌 Gemini session closed, cleared reference
DEBUG:websockets.client:< CLOSE 1000 (OK)
DEBUG:websockets.client:= connection is CLOSED
... (reconnecting in 2s)
✅ Gemini Live session started
📡 Listening for Gemini responses...
```

### During Reconnection (Race Condition)
```
⚠️  Gemini receive loop ended - reconnecting...
🔌 Connection closed while sending audio (will reconnect)  [DEBUG level]
🔌 Connection closed while sending audio (will reconnect)  [DEBUG level]
... (reconnecting)
✅ Gemini Live session started
```

---

## When to Worry

### ⚠️ Real Issues (Check These)

1. **Repeated authentication failures:**
   ```
   ❌ PERMISSIONS ERROR: Service account lacks required IAM permissions
   ```
   - **Action:** Check credentials and IAM permissions

2. **Continuous reconnection failures:**
   ```
   ❌ Error in Gemini session: [error] - reconnecting in 2s...
   ❌ Error in Gemini session: [error] - reconnecting in 2s...
   ... (repeats indefinitely)
   ```
   - **Action:** Check network connectivity, API quotas, service status

3. **No responses after reconnection:**
   - Session connects but never receives responses
   - **Action:** Check Gemini API status, model availability

### ✅ Normal Behavior (Ignore These)

1. **Occasional reconnections:**
   - Happens every few minutes to hours
   - Auto-recovers within 2 seconds
   - No user impact

2. **"Failed to send audio" during reconnection:**
   - Only happens during the brief reconnection window
   - Audio is buffered and will be sent after reconnect
   - Now logged at DEBUG level (less noisy)

3. **WebSocket close codes 1000:**
   - Normal closure, not an error
   - Part of healthy connection lifecycle

---

## Reducing Log Noise

### Option 1: Adjust Log Levels
```python
# In your code or environment
logging.getLogger('websockets').setLevel(logging.WARNING)  # Hide WebSocket DEBUG logs
logging.getLogger('__mp_main__').setLevel(logging.INFO)    # Hide DEBUG logs
```

### Option 2: Filter Specific Messages
```python
# Filter out connection closed warnings during reconnection
class ConnectionClosedFilter(logging.Filter):
    def filter(self, record):
        if "Connection closed while sending" in record.getMessage():
            return False  # Don't log this
        return True

logger.addFilter(ConnectionClosedFilter())
```

### Option 3: Use Structured Logging
- Use JSON logging with log levels
- Filter by level in your log aggregation system
- Set appropriate thresholds for alerts

---

## Summary

**Most warnings are normal:**
- Reconnections happen automatically
- Race conditions during reconnection are harmless
- WebSocket closures are part of normal operation

**What to monitor:**
- Authentication errors (real problems)
- Continuous reconnection failures (network/API issues)
- No responses after connection (service issues)

**The agent is designed to:**
- Auto-reconnect on connection drops
- Handle race conditions gracefully
- Continue operating during brief disconnections
- Log everything for debugging (hence the verbosity)

