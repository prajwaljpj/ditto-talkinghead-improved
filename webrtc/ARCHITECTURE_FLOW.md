# Gemini + Ditto Architecture Flow Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Browser["🌐 Browser Client"]
        Mic[🎤 Microphone Input]
        VideoDisplay[📺 Video Display]
        UI[💻 Web UI<br/>gemini_chat.html]
    end

    subgraph WebRTC["🔌 WebRTC Layer"]
        AudioOut[Audio Out Stream]
        VideoIn[Video In Stream]
        Signaling[WebSocket Signaling]
    end

    subgraph Server["🖥️ Python Server"]
        GeminiProc[🤖 GeminiProcessor<br/>ASR + LLM + TTS]
        DittoProc[🎭 DittoProcessor<br/>Talking Head]
        IdleProc[😴 IdleAnimator<br/>Subtle Movements]
    end

    subgraph External["☁️ External APIs"]
        GeminiAPI[Gemini Live API<br/>gemini-2.0-flash-exp]
    end

    Mic -->|User Audio| AudioOut
    AudioOut -->|WebRTC| GeminiProc

    GeminiProc -->|User Audio| GeminiAPI
    GeminiAPI -->|TTS Audio| GeminiProc

    GeminiProc -->|Audio<br/>TTS or Silence| DittoProc
    DittoProc -->|Video Frames| IdleProc
    IdleProc -->|Final Video| VideoIn

    VideoIn -->|WebRTC| VideoDisplay
    UI -.->|Controls| Signaling

    style GeminiProc fill:#667eea,color:#fff
    style DittoProc fill:#764ba2,color:#fff
    style GeminiAPI fill:#4285f4,color:#fff
```

## 2. Audio Flow Decision Tree

```mermaid
graph TD
    Start([User Audio Frame Arrives]) --> GeminiProc{GeminiProcessor<br/>Receives Frame}

    GeminiProc --> SendToAPI[Send to Gemini API<br/>for ASR + LLM]

    GeminiProc --> CheckState{Is Avatar<br/>Currently<br/>Speaking?}

    CheckState -->|YES| WaitForTTS[Avatar is speaking<br/>TTS already queued]
    CheckState -->|NO| SendSilence[Generate Silence Frame<br/>b'\x00' * frame_length]

    SendSilence --> PushSilence[Push Silence to<br/>DittoProcessor]

    SendToAPI --> GeminiResponse{Gemini<br/>Response<br/>Ready?}

    GeminiResponse -->|Text| ExtractEmotion[Extract Emotion<br/>from Text]
    GeminiResponse -->|Audio| QueueTTS[Queue TTS Audio<br/>for Output]

    ExtractEmotion --> UpdateEmotion[Update Avatar<br/>Emotion State]

    QueueTTS --> PushTTS[Push TTS Audio to<br/>DittoProcessor]

    PushSilence --> DittoProcess[DittoProcessor<br/>Generates Video]
    PushTTS --> DittoProcess
    WaitForTTS --> DittoProcess

    DittoProcess --> Output([Video Frame Output<br/>to Browser])

    style CheckState fill:#fbbf24,color:#000
    style SendSilence fill:#10b981,color:#fff
    style PushTTS fill:#3b82f6,color:#fff
    style DittoProcess fill:#764ba2,color:#fff
```

## 3. Three Audio States in Detail

```mermaid
stateDiagram-v2
    [*] --> Idle: Session Starts

    state "😴 IDLE STATE" as Idle {
        [*] --> NoAudio: No Input
        NoAudio --> SendingSilence: Continuous
        SendingSilence --> DittoIdle: Silence Frames
        DittoIdle --> IdleAnimation: Subtle Movements
        IdleAnimation --> [*]
    }

    state "👂 LISTENING STATE" as Listening {
        [*] --> UserSpeaking: User Audio Detected
        UserSpeaking --> ToGemini: Send to API
        UserSpeaking --> SendingSilence2: Also Send Silence
        SendingSilence2 --> DittoListening: Silence Frames
        DittoListening --> ListeningPose: Idle/Listening Pose
        ToGemini --> WaitingResponse: Processing
        WaitingResponse --> [*]
    }

    state "🗣️ SPEAKING STATE" as Speaking {
        [*] --> GeminiResponse: TTS Audio Ready
        GeminiResponse --> SendingTTS: Queue TTS
        SendingTTS --> DittoSpeaking: TTS Audio Frames
        DittoSpeaking --> LipSync: Lip-Synced Video
        LipSync --> CheckComplete{More Audio?}
        CheckComplete --> SendingTTS: Yes
        CheckComplete --> [*]: No
    }

    Idle --> Listening: User Starts Speaking
    Idle --> Speaking: Gemini Starts Speaking

    Listening --> Speaking: Gemini Responds
    Listening --> Idle: User Stops

    Speaking --> Listening: User Interrupts
    Speaking --> Idle: Speech Complete

    note right of Idle
        Audio to Ditto: SILENCE
        Avatar: Subtle idle movements
        VAD: No activity
    end note

    note right of Listening
        Audio to Ditto: SILENCE
        Avatar: Listening pose
        VAD: User detected
    end note

    note right of Speaking
        Audio to Ditto: GEMINI TTS
        Avatar: Lip-synced speech
        VAD: Avatar active
    end note
```

## 4. Detailed Pipeline Flow (Frame by Frame)

```mermaid
sequenceDiagram
    participant Browser
    participant WebRTC
    participant GeminiProc as GeminiProcessor
    participant GeminiAPI as Gemini API
    participant DittoProc as DittoProcessor
    participant StreamSDK as Ditto StreamSDK
    participant IdleAnim as IdleAnimator

    Note over Browser,IdleAnim: INITIALIZATION PHASE
    Browser->>WebRTC: Connect
    WebRTC->>GeminiProc: StartFrame
    GeminiProc->>GeminiAPI: Initialize Session
    GeminiAPI-->>GeminiProc: Session Ready
    GeminiProc->>DittoProc: StartFrame
    DittoProc->>StreamSDK: setup(source_image)

    Note over Browser,IdleAnim: USER SPEAKING PHASE
    loop Every 20ms (audio frame)
        Browser->>WebRTC: AudioRawFrame (user audio)
        WebRTC->>GeminiProc: AudioRawFrame

        par Send to Gemini
            GeminiProc->>GeminiAPI: User Audio (PCM16)
        and Send Silence to Ditto
            GeminiProc->>GeminiProc: is_speaking = False
            GeminiProc->>GeminiProc: Generate Silence Frame
            GeminiProc->>DittoProc: AudioRawFrame (SILENCE)
            DittoProc->>StreamSDK: run_chunk(silence_audio)
            StreamSDK-->>DittoProc: Video Frame (idle pose)
            DittoProc->>IdleAnim: OutputImageRawFrame
            IdleAnim-->>WebRTC: Video Frame
            WebRTC-->>Browser: Display Video
        end
    end

    Note over Browser,IdleAnim: GEMINI PROCESSING PHASE
    GeminiAPI->>GeminiAPI: ASR (speech → text)
    GeminiAPI->>GeminiAPI: LLM (generate response)
    GeminiAPI->>GeminiAPI: TTS (text → audio)

    Note over Browser,IdleAnim: AVATAR SPEAKING PHASE
    GeminiAPI-->>GeminiProc: Text Response
    GeminiProc->>GeminiProc: Detect Emotion
    GeminiProc->>DittoProc: update_emotion("happy")

    loop TTS Audio Chunks
        GeminiAPI-->>GeminiProc: Audio Chunk (PCM16)
        GeminiProc->>GeminiProc: is_speaking = True
        GeminiProc->>GeminiProc: Queue Audio
        GeminiProc->>DittoProc: AudioRawFrame (TTS audio)
        DittoProc->>StreamSDK: run_chunk(tts_audio)
        StreamSDK->>StreamSDK: Audio2Motion → MotionStitch
        StreamSDK->>StreamSDK: Warp3D → Decode → PutBack
        StreamSDK-->>DittoProc: Video Frame (lip-synced)
        DittoProc->>IdleAnim: OutputImageRawFrame
        IdleAnim-->>WebRTC: Video Frame
        WebRTC-->>Browser: Display Video
    end

    GeminiAPI-->>GeminiProc: Turn Complete
    GeminiProc->>GeminiProc: is_speaking = False

    Note over Browser,IdleAnim: BACK TO IDLE/LISTENING
    loop Continuous
        Browser->>WebRTC: AudioRawFrame (silence/user)
        WebRTC->>GeminiProc: AudioRawFrame
        GeminiProc->>DittoProc: AudioRawFrame (SILENCE)
        DittoProc->>StreamSDK: run_chunk(silence)
        StreamSDK-->>DittoProc: Video Frame (idle)
        DittoProc->>IdleAnim: OutputImageRawFrame
        IdleAnim-->>WebRTC: Video Frame
        WebRTC-->>Browser: Display Video
    end
```

## 5. Emotion Detection and Update Flow

```mermaid
graph LR
    subgraph GeminiResponse["Gemini Text Response"]
        TextResp[Text: 'I'm so happy<br/>to help you!']
    end

    subgraph ConversationManager["ConversationManager"]
        DetectEmo[Detect Emotion<br/>Keyword Matching]
        Keywords["Keywords:<br/>happy → HAPPY<br/>sad → SAD<br/>angry → ANGRY"]
        EmoState[Update Emotion State<br/>with Smoothing]
    end

    subgraph DittoUpdate["Ditto Emotion Update"]
        MapEmo[Map Emotion to Code<br/>HAPPY → 0<br/>NEUTRAL → 4<br/>SAD → 2]
        UpdateSDK[Update StreamSDK<br/>emo parameter]
        FaceExpr[Facial Expression<br/>Changes]
    end

    TextResp --> DetectEmo
    DetectEmo --> Keywords
    Keywords --> EmoState
    EmoState -->|EmotionType.HAPPY| MapEmo
    MapEmo -->|Code: 0| UpdateSDK
    UpdateSDK --> FaceExpr

    style DetectEmo fill:#fbbf24,color:#000
    style EmoState fill:#10b981,color:#fff
    style FaceExpr fill:#764ba2,color:#fff
```

## 6. Interruption Handling Flow

```mermaid
flowchart TD
    Start([Audio Frame Arrives]) --> Process[GeminiProcessor<br/>Receives Frame]

    Process --> CalcVAD[Calculate Audio Level<br/>np.abs(audio).mean]

    CalcVAD --> CheckVAD{Audio Level ><br/>VAD Threshold?}

    CheckVAD -->|YES| UserSpeaking[is_user_speaking = True]
    CheckVAD -->|NO| UserSilent[is_user_speaking = False]

    UserSpeaking --> CheckInterrupt{is_speaking<br/>Avatar talking?}

    CheckInterrupt -->|YES| Interrupt[🚨 INTERRUPTION!<br/>is_speaking = False]
    CheckInterrupt -->|NO| Normal[Continue Normal Flow]

    Interrupt --> StopTTS[Clear TTS Queue]
    StopTTS --> SendSilence[Send Silence to Ditto]

    UserSilent --> CheckEndTurn{was_speaking<br/>before?}
    CheckEndTurn -->|YES| EndTurn[Send end_of_turn<br/>to Gemini]
    CheckEndTurn -->|NO| Continue[Continue]

    Normal --> SendSilence
    SendSilence --> Output([Output to Ditto])
    EndTurn --> SendSilence
    Continue --> SendSilence

    style Interrupt fill:#ef4444,color:#fff
    style CheckInterrupt fill:#fbbf24,color:#000
    style StopTTS fill:#f97316,color:#fff
```

## 7. Complete Data Flow with All Components

```mermaid
graph TB
    subgraph Client["🌐 Browser Client"]
        direction TB
        Mic[🎤 Microphone<br/>48kHz PCM]
        Video[📺 Video Display<br/>25fps]
        Transcript[💬 Transcript UI]
        EmotionUI[😊 Emotion Badge]
        Stats[📊 Statistics]
    end

    subgraph Processors["🔄 Processing Pipeline"]
        direction TB

        subgraph GP["GeminiProcessor"]
            direction LR
            GP1[Receive User Audio]
            GP2[VAD Detection]
            GP3{Is Avatar<br/>Speaking?}
            GP4[Generate Silence]
            GP5[Forward TTS]

            GP1 --> GP2
            GP2 --> GP3
            GP3 -->|No| GP4
            GP3 -->|Yes| GP5
        end

        subgraph CM["ConversationManager"]
            direction LR
            CM1[Store Turn]
            CM2[Detect Emotion]
            CM3[Update Context]

            CM1 --> CM2
            CM2 --> CM3
        end

        subgraph DP["DittoProcessor"]
            direction LR
            DP1[Receive Audio<br/>TTS or Silence]
            DP2[Audio2Motion]
            DP3[Motion Processing]
            DP4[Video Generation]

            DP1 --> DP2
            DP2 --> DP3
            DP3 --> DP4
        end

        subgraph IA["IdleAnimator"]
            IA1[Add Subtle<br/>Movements]
        end
    end

    subgraph APIs["☁️ External Services"]
        direction TB
        GA[Gemini Live API]
        GA1[ASR]
        GA2[LLM]
        GA3[TTS]

        GA --> GA1
        GA1 --> GA2
        GA2 --> GA3
    end

    Mic -->|User Audio| GP1

    GP1 -.->|User Audio| GA
    GA3 -.->|TTS Audio| GP5
    GA2 -.->|Text| CM1

    GP4 -->|Silence Frames| DP1
    GP5 -->|TTS Frames| DP1

    CM2 -->|Emotion| DP

    DP4 -->|Video Frames| IA1
    IA1 -->|Final Video| Video

    CM3 -.->|Transcript| Transcript
    CM2 -.->|Emotion| EmotionUI
    GP2 -.->|Audio Levels| Stats

    style GP fill:#667eea,color:#fff
    style DP fill:#764ba2,color:#fff
    style GA fill:#4285f4,color:#fff
    style CM fill:#10b981,color:#fff
```

## 8. Audio Buffer Management

```mermaid
graph LR
    subgraph Input["Input Pipeline"]
        direction TB
        Browser[Browser Audio<br/>48kHz, 20ms chunks]
        Resample[Resample to 16kHz<br/>scipy.signal.resample]
        VAD[Voice Activity<br/>Detection]
    end

    subgraph GeminiQueue["Gemini Queue"]
        direction TB
        UserQ[User Audio Queue<br/>Send to Gemini]
        TTSQ[TTS Response Queue<br/>From Gemini]
    end

    subgraph DittoQueue["Ditto Queue"]
        direction TB
        AudioQ[Audio Queue<br/>Silence or TTS]
        Buffer[Buffer Management<br/>640 samples/chunk]
        Chunk[Chunking<br/>chunksize=(3,5,2)]
    end

    subgraph Output["Output Pipeline"]
        direction TB
        VideoQ[Video Frame Queue]
        Encode[H264 Encoding]
        Stream[WebRTC Stream]
    end

    Browser --> Resample
    Resample --> VAD
    VAD --> UserQ

    UserQ -.->|Network| TTSQ

    TTSQ --> AudioQ
    VAD -->|Silence when<br/>not speaking| AudioQ

    AudioQ --> Buffer
    Buffer --> Chunk
    Chunk --> VideoQ

    VideoQ --> Encode
    Encode --> Stream

    style VAD fill:#fbbf24,color:#000
    style AudioQ fill:#3b82f6,color:#fff
    style VideoQ fill:#764ba2,color:#fff
```

## Legend

- 🌐 Browser/Client components
- 🔌 WebRTC/Network layer
- 🤖 AI/ML processors
- 🎭 Video generation
- ☁️ External APIs
- 💬 User interface
- 📊 Monitoring/Stats

## Key Takeaways

1. **Audio Always Flows**: Ditto NEVER freezes because it always receives audio (TTS or silence)
2. **Three States**: Idle (silence), Listening (user speaks, silence), Speaking (Gemini TTS)
3. **Parallel Processing**: User audio → Gemini AND silence → Ditto (simultaneously)
4. **Emotion Integration**: Text responses trigger emotion detection → update Ditto expression
5. **Interruption Handling**: VAD detects user speech → stops avatar → switches to silence frames
6. **Frame Synchronization**: Audio chunks (40ms) → Video frames (25fps) maintained by StreamSDK

