import time
import threading
from collections import deque
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TranscriptionSegment:
    text: str
    timestamp: datetime
    speaker_id: Optional[str] = None
    confidence: float = 1.0
    is_final: bool = False


@dataclass
class ConversationContext:
    segments: List[TranscriptionSegment] = field(default_factory=list)
    current_framework_step: str = ""
    framework_name: str = ""
    meeting_goal: str = ""
    participants: List[str] = field(default_factory=list)
    last_ai_response_time: Optional[datetime] = None


class TranscriptionHandler:
    def __init__(self, response_callback: Callable[[str], None]):
        self.response_callback = response_callback
        self.context = ConversationContext()
        self.transcription_buffer = deque(maxlen=50)  # Keep last 50 segments
        self.pending_segments = []
        self.last_transcription_time = None
        self.silence_threshold = 5.0  # seconds
        self.min_response_interval = 15.0  # minimum seconds between AI responses

        # Start background thread for processing
        self.processing_thread = threading.Thread(target=self._process_transcriptions, daemon=True)
        self.processing_thread.start()

    def add_transcription(self, text: str, is_final: bool = True, speaker_id: str = None):
        """Add new transcription from Google STT"""
        if not text.strip():
            return

        segment = TranscriptionSegment(
            text=text.strip(),
            timestamp=datetime.now(),
            speaker_id=speaker_id,
            is_final=is_final
        )

        self.transcription_buffer.append(segment)
        self.context.segments.append(segment)
        self.last_transcription_time = time.time()

        print(f"[TRANSCRIPTION] {segment.text}")

    def set_framework_context(self, framework_name: str, current_step: str, meeting_goal: str):
        """Set the current framework context"""
        self.context.framework_name = framework_name
        self.context.current_framework_step = current_step
        self.context.meeting_goal = meeting_goal

    def get_recent_conversation(self, minutes: int = 5) -> List[TranscriptionSegment]:
        """Get conversation from last N minutes"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        return [
            segment for segment in self.context.segments
            if segment.timestamp.timestamp() > cutoff_time
        ]

    def should_trigger_ai_response(self) -> bool:
        """Determine if we should send to LLM for analysis"""
        if not self.last_transcription_time:
            return False

        # Check if enough silence has passed
        silence_duration = time.time() - self.last_transcription_time
        if silence_duration < self.silence_threshold:
            return False

        # Check minimum interval between AI responses
        if self.context.last_ai_response_time:
            time_since_last_response = (datetime.now() - self.context.last_ai_response_time).total_seconds()
            if time_since_last_response < self.min_response_interval:
                return False

        # Check if we have meaningful content
        recent_segments = self.get_recent_conversation(minutes=2)
        if len(recent_segments) < 1:  # Need at least some conversation
            return False

        return True

    def _process_transcriptions(self):
        """Background thread to process transcriptions and trigger AI responses"""
        while True:
            try:
                if self.should_trigger_ai_response():
                    recent_conversation = self.get_recent_conversation(minutes=3)
                    conversation_text = " ".join([seg.text for seg in recent_conversation])

                    if len(conversation_text.strip()) > 100:  # Minimum meaningful content
                        print(f"[PROCESSING] Sending to LLM: {conversation_text[:100]}...")
                        self.response_callback(conversation_text)
                        self.context.last_ai_response_time = datetime.now()

                time.sleep(1)  # Check every second
            except Exception as e:
                print(f"Error in transcription processing: {e}")
                time.sleep(5)

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation for LLM context"""
        recent_segments = self.get_recent_conversation(minutes=10)
        if not recent_segments:
            return ""

        conversation_lines = []
        for segment in recent_segments[-15:]:  # Last 15 segments
            timestamp = segment.timestamp.strftime("%H:%M:%S")
            speaker = segment.speaker_id or "Participant"
            conversation_lines.append(f"[{timestamp}] {speaker}: {segment.text}")

        return "\n".join(conversation_lines)