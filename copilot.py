"""
Sales Copilot Main Module.

This module coordinates the audio capture, speech recognition,
and provides the main application logic for the sales copilot.
"""
import sys
import threading
import time
import traceback
from typing import List, Dict, Any, Optional

# Import from our modules
from audio_capture import MicrophoneStream, SystemAudioStream, get_current_time, STREAMING_LIMIT
from framework_engine import FrameworkEngine
from llm_facilitator import LLMFacilitator
from speech_recognition import SpeechRecognizer, listen_print_loop, YELLOW
from transcription_handler import TranscriptionHandler

# Constants
MICROPHONE_SAMPLE_RATE = 16000
MICROPHONE_CHUNK_SIZE = int(MICROPHONE_SAMPLE_RATE / 10)  # 100ms

SYSTEM_AUDIO_SAMPLE_RATE = 48000
SYSTEM_AUDIO_CHUNK_SIZE = 150  # Small chunk size for system audio

# Global components
framework_engine = FrameworkEngine()
facilitator = None
transcription_handler = None
current_meeting_goal = ""


def on_ai_response(conversation_text: str):
    """Callback function when transcription handler wants AI analysis"""
    global facilitator, current_meeting_goal

    if not facilitator:
        return

    try:
        ai_message = facilitator.process_conversation(conversation_text, current_meeting_goal)
        if ai_message:
            print(f"[AI RESPONSE] {ai_message}")
    except Exception as e:
        print(f"Error processing AI response: {e}")

class SalesCopilot:
    """Main Sales Copilot class that orchestrates the entire process."""

    def __init__(self,
                 credentials_path: str,
                 language_code: str = "en-US",
                 streaming_limit: int = STREAMING_LIMIT):
        """Initialize the Sales Copilot.

        Args:
            credentials_path: Path to Google Cloud credentials.
            language_code: Language code for speech recognition.
            streaming_limit: Time limit for streaming in milliseconds.
        """
        self.credentials_path = credentials_path
        self.language_code = language_code
        self.streaming_limit = streaming_limit

        # Initialize recognizers
        self.mic_recognizer = SpeechRecognizer(
            sample_rate=MICROPHONE_SAMPLE_RATE,
            language_code=language_code,
            streaming_limit=streaming_limit,
            credentials_path=credentials_path
        )

        self.system_recognizer = SpeechRecognizer(
            sample_rate=SYSTEM_AUDIO_SAMPLE_RATE,
            language_code=language_code,
            streaming_limit=streaming_limit,
            credentials_path=credentials_path
        )

        # Transcription storage
        self.all_transcriptions = []

        # Add lock for thread-safe transcription processing
        self.transcription_lock = threading.Lock()
        global facilitator, transcription_handler, current_meeting_goal

        framework_name = '5_whys'
        meeting_goal = 'Solve a problem using structured approach'
        gemini_api_key = 'AIzaSyDlUCOnyKPDe281b6WvJwCIOnEjhqKf8j8'

        try:
            # Initialize components
            facilitator = LLMFacilitator(gemini_api_key, framework_engine)
            transcription_handler = TranscriptionHandler(on_ai_response)
            current_meeting_goal = meeting_goal

            # Set framework context
            if not facilitator.set_meeting_context(framework_name, meeting_goal):
                print ({'error': f'Framework {framework_name} not found'})

            # Set context in transcription handler
            transcription_handler.set_framework_context(framework_name, "problem_definition", meeting_goal)

            step_info = facilitator.get_current_step_info()

        except Exception as e:
            print ({'error': f'Failed to start meeting: {str(e)}'})

    def process_microphone(self):
        """Process audio from microphone and transcribe it."""
        mic_manager = MicrophoneStream(MICROPHONE_SAMPLE_RATE, MICROPHONE_CHUNK_SIZE)
        global transcription_handler
        with mic_manager as stream:
            while not stream.closed:
                sys.stdout.write(YELLOW)
                sys.stdout.write(
                    "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": MIC NEW REQUEST\n"
                )

                stream.audio_input = []

                # Get responses from the speech recognizer
                responses = self.mic_recognizer.process_audio_stream(stream)

                # Process the responses
                listen_print_loop(responses, stream, transcription_handler)

                # Update stream state for next iteration
                if stream.result_end_time > 0:
                    stream.final_request_end_time = stream.is_final_end_time
                stream.result_end_time = 0
                stream.last_audio_input = []
                stream.last_audio_input = stream.audio_input
                stream.audio_input = []
                stream.restart_counter = stream.restart_counter + 1

                if not stream.last_transcript_was_final:
                    sys.stdout.write("\n")
                stream.new_stream = True

                # Small yield to prevent blocking
                time.sleep(0.1)

    def process_system_audio(self):
        """Process audio from system output and transcribe it."""
        try:
            sys_manager = SystemAudioStream(SYSTEM_AUDIO_SAMPLE_RATE, SYSTEM_AUDIO_CHUNK_SIZE)
            global transcription_handler

            with sys_manager as stream:
                while not stream.closed:
                    sys.stdout.write(YELLOW)
                    sys.stdout.write(
                        "\n" + str(STREAMING_LIMIT * stream.restart_counter) + ": SYS NEW REQUEST\n"
                    )

                    stream.audio_input = []

                    # Get responses from the speech recognizer
                    responses = self.system_recognizer.process_audio_stream(stream)

                    # Process the responses
                    listen_print_loop(responses, stream, transcription_handler)

                    # Update stream state for next iteration
                    if stream.result_end_time > 0:
                        stream.final_request_end_time = stream.is_final_end_time
                    stream.result_end_time = 0
                    stream.last_audio_input = []
                    stream.last_audio_input = stream.audio_input
                    stream.audio_input = []
                    stream.restart_counter = stream.restart_counter + 1

                    if not stream.last_transcript_was_final:
                        sys.stdout.write("\n")
                    stream.new_stream = True

                    # Small yield to prevent blocking
                    time.sleep(0.1)

        except Exception as e:
            print(f"Error in system audio processing: {e}")
            raise

    def process_transcription(self, transcription: str, source: str):
        """Process a single transcription as it comes in.

        Args:
            transcription: The transcribed text.
            source: The source of the transcription ('mic' or 'system').
        """
        with self.transcription_lock:
            # Store the transcription
            timestamp = get_current_time()
            self.all_transcriptions.append({
                'text': transcription,
                'source': source,
                'timestamp': timestamp
            })

            # Print with a consistent format for debugging
            formatted_time = time.strftime('%H:%M:%S', time.localtime(timestamp / 1000))
            print(f"[{formatted_time}] {source.upper()}: {transcription}")

            # TODO: Send to LLM for real-time analysis
            # This is where you would call your LLM processing

    def run(self):
        """Run the sales copilot with concurrent audio processing."""
        print("Starting Sales Copilot")
        print("======================")
        print('Listening from microphone and system audio simultaneously.')
        print('Say "Quit" or "Exit" to stop.\n')

        # Create threads for each audio processing function
        mic_thread = threading.Thread(target=self.process_microphone, name="MicrophoneThread")
        system_audio_thread = threading.Thread(target=self.process_system_audio, name="SystemAudioThread")

        # Set as daemon threads, so they exit when the main program exits
        mic_thread.daemon = True
        system_audio_thread.daemon = True

        # Start both threads
        system_audio_thread.start()
        mic_thread.start()

        try:
            # Keep the main thread alive until interrupted
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down Sales Copilot...")




