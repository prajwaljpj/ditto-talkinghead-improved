"""
Conversation Context Manager for Gemini Live API integration.

Manages multi-turn conversation history, emotional state tracking,
and context window management for natural conversational interactions.
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class EmotionType(Enum):
    """Emotion types mapped to Ditto emotion codes."""
    NEUTRAL = "neu"
    HAPPY = "hap"
    SAD = "sad"
    ANGRY = "ang"
    SURPRISED = "sur"


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user" or "model"
    text: str
    audio_data: Optional[bytes] = None
    emotion: EmotionType = EmotionType.NEUTRAL
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Manages conversation state, context, and emotional flow.

    Features:
    - Multi-turn conversation history with sliding window
    - Emotion tracking and smoothing
    - Session management with timeouts
    - Context serialization for persistence
    """

    def __init__(
        self,
        max_history: int = 20,
        context_window: int = 10,
        session_timeout: float = 300.0,  # 5 minutes
        emotion_smoothing: float = 0.7,  # 0-1, higher = smoother transitions
    ):
        """
        Initialize conversation manager.

        Args:
            max_history: Maximum number of turns to store
            context_window: Number of recent turns to include in context
            session_timeout: Timeout in seconds for session expiry
            emotion_smoothing: Smoothing factor for emotion transitions
        """
        self.max_history = max_history
        self.context_window = context_window
        self.session_timeout = session_timeout
        self.emotion_smoothing = emotion_smoothing

        # Conversation state
        self.history: List[ConversationTurn] = []
        self.current_emotion = EmotionType.NEUTRAL
        self.last_activity = time.time()

        # Session metadata
        self.session_id: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def add_turn(
        self,
        role: str,
        text: str,
        audio_data: Optional[bytes] = None,
        emotion: Optional[EmotionType] = None,
        **metadata
    ) -> ConversationTurn:
        """
        Add a new turn to the conversation.

        Args:
            role: "user" or "model"
            text: Transcribed or generated text
            audio_data: Raw audio bytes (optional)
            emotion: Detected/generated emotion
            **metadata: Additional metadata for this turn

        Returns:
            The created ConversationTurn
        """
        async with self._lock:
            # Auto-detect emotion if not provided (simple sentiment analysis)
            if emotion is None and role == "model":
                emotion = self._detect_emotion(text)
            elif emotion is None:
                emotion = EmotionType.NEUTRAL

            # Create turn
            turn = ConversationTurn(
                role=role,
                text=text,
                audio_data=audio_data,
                emotion=emotion,
                metadata=metadata
            )

            # Add to history
            self.history.append(turn)

            # Trim history if needed
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            # Update emotional state
            if role == "model":
                self._update_emotion(emotion)

            # Update activity timestamp
            self.last_activity = time.time()

            return turn

    async def get_context(self, include_system_prompt: bool = True) -> List[Dict[str, str]]:
        """
        Get recent conversation context for Gemini API.

        Args:
            include_system_prompt: Whether to include system instructions

        Returns:
            List of message dicts suitable for Gemini API
        """
        async with self._lock:
            # Get recent turns within context window
            recent_turns = self.history[-self.context_window:]

            # Convert to Gemini message format
            messages = []

            if include_system_prompt:
                messages.append({
                    "role": "system",
                    "content": self._get_system_prompt()
                })

            for turn in recent_turns:
                messages.append({
                    "role": turn.role,
                    "content": turn.text
                })

            return messages

    async def get_current_emotion(self) -> EmotionType:
        """Get the current emotional state."""
        async with self._lock:
            return self.current_emotion

    async def reset(self):
        """Reset conversation state."""
        async with self._lock:
            self.history.clear()
            self.current_emotion = EmotionType.NEUTRAL
            self.last_activity = time.time()
            print(f"[ConversationManager] Reset conversation (session: {self.session_id})")

    async def is_expired(self) -> bool:
        """Check if session has expired due to inactivity."""
        return (time.time() - self.last_activity) > self.session_timeout

    def _detect_emotion(self, text: str) -> EmotionType:
        """
        Simple emotion detection from text.

        This is a basic implementation using keyword matching.
        For production, consider using a proper sentiment analysis model.
        """
        text_lower = text.lower()

        # Keyword-based emotion detection
        happy_keywords = ["happy", "great", "wonderful", "excited", "joy", "love", "haha", "lol"]
        sad_keywords = ["sad", "sorry", "unfortunately", "regret", "disappointed"]
        angry_keywords = ["angry", "frustrated", "annoyed", "ridiculous", "terrible"]
        surprised_keywords = ["wow", "amazing", "incredible", "surprised", "unbelievable"]

        # Count keyword matches
        happy_score = sum(1 for kw in happy_keywords if kw in text_lower)
        sad_score = sum(1 for kw in sad_keywords if kw in text_lower)
        angry_score = sum(1 for kw in angry_keywords if kw in text_lower)
        surprised_score = sum(1 for kw in surprised_keywords if kw in text_lower)

        # Determine dominant emotion
        scores = [
            (happy_score, EmotionType.HAPPY),
            (sad_score, EmotionType.SAD),
            (angry_score, EmotionType.ANGRY),
            (surprised_score, EmotionType.SURPRISED),
        ]

        max_score, emotion = max(scores, key=lambda x: x[0])

        # Return neutral if no clear emotion detected
        return emotion if max_score > 0 else EmotionType.NEUTRAL

    def _update_emotion(self, new_emotion: EmotionType):
        """
        Update current emotion with smoothing.

        Applies exponential smoothing to avoid abrupt emotion changes.
        """
        # For simplicity, we'll use a weighted approach
        # In a more sophisticated system, you'd blend emotion intensities

        if self.emotion_smoothing < 0.5:
            # Low smoothing - change quickly
            self.current_emotion = new_emotion
        elif self.current_emotion == new_emotion:
            # Same emotion - reinforce it
            self.current_emotion = new_emotion
        else:
            # High smoothing - only change if emotion is strong
            # For now, we'll keep the current emotion unless it's neutral
            if self.current_emotion == EmotionType.NEUTRAL:
                self.current_emotion = new_emotion
            # Otherwise, keep current emotion (high smoothing)

    def _get_system_prompt(self) -> str:
        """Get system prompt for Gemini API."""
        return """You are a friendly and expressive conversational AI avatar.

Guidelines:
- Be natural, conversational, and engaging
- Keep responses concise (1-3 sentences typically)
- Show emotion through your word choice
- Ask follow-up questions to maintain engagement
- Be helpful and informative while staying personable

Remember: Your responses will be spoken aloud by an animated avatar,
so prioritize clarity and natural speech patterns."""

    async def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state."""
        async with self._lock:
            return {
                "session_id": self.session_id,
                "turn_count": len(self.history),
                "current_emotion": self.current_emotion.value,
                "last_activity": self.last_activity,
                "is_expired": await self.is_expired(),
                "recent_turns": [
                    {
                        "role": turn.role,
                        "text": turn.text[:50] + "..." if len(turn.text) > 50 else turn.text,
                        "emotion": turn.emotion.value,
                        "timestamp": turn.timestamp
                    }
                    for turn in self.history[-5:]  # Last 5 turns
                ]
            }
