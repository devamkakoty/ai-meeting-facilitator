"""
Speech Recognition Module for the Copilot.

Handles the interaction with Google Cloud Speech-to-Text API.
"""
import os
import sys
import time
from google.cloud import speech
from config import STREAMING_LIMIT
from transcription_handler import TranscriptionHandler

# Color constants for display
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))

class SpeechRecognizer:
    """Handles speech recognition using Google Cloud Speech-to-Text API."""

    def __init__(self,
                 sample_rate: int,
                 language_code: str = "en-US",
                 streaming_limit: int = 240000,
                 credentials_path: str = None):
        """Initialize the speech recognizer.

        Args:
            sample_rate: The audio sample rate in Hz.
            language_code: The language code for recognition.
            streaming_limit: The time limit for streaming in milliseconds.
            credentials_path: Path to the Google Cloud credentials file.
        """
        self.sample_rate = sample_rate
        self.language_code = language_code
        self.streaming_limit = streaming_limit

        # Set Google Cloud credentials if provided
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        # Initialize the client
        self.client = speech.SpeechClient()

    def create_recognition_config(self, encoding=None, max_alternatives=1):
        """Create a recognition config for the speech API.

        Args:
            encoding: The audio encoding type.
            max_alternatives: Maximum number of alternatives to return.

        Returns:
            RecognitionConfig: The recognition configuration.
        """
        if encoding is None:
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16

        return speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            max_alternatives=max_alternatives,
        )

    def create_streaming_config(self, recognition_config=None, interim_results=True):
        """Create a streaming recognition config.

        Args:
            recognition_config: The base recognition config.
            interim_results: Whether to return interim results.

        Returns:
            StreamingRecognitionConfig: The streaming config.
        """
        if recognition_config is None:
            recognition_config = self.create_recognition_config()

        return speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=interim_results
        )

    def process_audio_stream(self, audio_stream):
        """Process audio from a stream and return transcription responses.

        Args:
            audio_stream: The audio stream to process.

        Returns:
            Generator of StreamingRecognizeResponse objects.
        """
        streaming_config = self.create_streaming_config()

        # Create a generator of requests from the audio stream
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_stream.generator()
        )

        # Get streaming recognition responses
        return self.client.streaming_recognize(streaming_config, requests)


def listen_print_loop(responses, stream, transcription_handler: TranscriptionHandler):
    """Iterates through server responses and prints them.

    Args:
        responses: The responses returned from the API.
        stream: The audio stream being processed.
        :param stream:
        :param responses:
        :param transcription_handler:
    """
    for response in responses:
        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
                stream.result_end_time
                - stream.bridging_offset
                + (STREAMING_LIMIT * stream.restart_counter)
        )

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        if result.is_final:
            # NEW: Send to meeting facilitator
            transcription_handler.add_transcription(
                text=transcript,
                is_final= True,
                speaker_id=None  # You can implement speaker identification if needed
            )
            #sys.stdout.write(GREEN)
            #sys.stdout.write("\033[K")
            #sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            # Flush stdout to ensure immediate display
            #sys.stdout.flush()

            #stream.is_final_end_time = stream.result_end_time
            #stream.last_transcript_was_final = True

        else:
            pass
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

            # Flush stdout to ensure immediate display
            sys.stdout.flush()

            stream.last_transcript_was_final = False