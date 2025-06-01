"""
Audio Capture Module for Copilot.

This module handles capturing audio from different sources (microphone and system audio).
"""
import queue
import time
import numpy as np
import pyaudio
import pyaudiowpatch as pyaudiow  # For system audio capture

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes


def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """
    return int(round(time.time() * 1000))


class BaseAudioStream:
    """Base class for audio stream handling."""

    def __init__(
            self,
            rate: int,
            chunk_size: int,
    ) -> None:
        """Create a base audio stream.

        Args:
            rate: The audio file's sampling rate.
            chunk_size: The audio file's chunk size.
        """
        self._rate = rate
        self.chunk_size = chunk_size
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True

    def __enter__(self) -> object:
        """Opens the stream.

        Returns:
            The stream object.
        """
        self.closed = False
        return self

    def __exit__(
            self,
            type: object,
            value: object,
            traceback: object,
    ) -> None:
        """Closes the stream and releases resources.

        Args:
            type: The exception type.
            value: The exception value.
            traceback: The exception traceback.
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
            self,
            in_data: object,
            *args: object,
            **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream into the buffer.

        Args:
            in_data: The audio data as a bytes object.
            args: Additional arguments.
            kwargs: Additional arguments.

        Returns:
            Tuple of (None, paContinue)
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> object:
        """Stream Audio from source to API and local buffer.

        Yields:
            Chunks of audio data.
        """
        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)


class MicrophoneStream(BaseAudioStream):
    """Opens a recording stream from microphone as a generator yielding the audio chunks."""

    def __init__(
            self,
            rate: int,
            chunk_size: int,
    ) -> None:
        """Creates a microphone stream.

        Args:
            rate: The audio file's sampling rate.
            chunk_size: The audio file's chunk size.
        """
        super().__init__(rate, chunk_size)
        self._num_channels = 1
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )


class SystemAudioStream(BaseAudioStream):
    """Opens a recording stream from system audio as a generator yielding the audio chunks."""

    def __init__(
            self,
            rate: int,
            chunk_size: int,
    ) -> None:
        """Creates a system audio stream.

        Args:
            rate: The audio file's sampling rate.
            chunk_size: The audio file's chunk size.
        """
        super().__init__(rate, chunk_size)
        self._num_channels = None
        self.default_speakers = None
        self._audio_interface = pyaudiow.PyAudio()
        self.initialize_speakers()
        self._audio_stream = self._audio_interface.open(
            format=pyaudiow.paInt16,
            channels=self.default_speakers["maxInputChannels"],
            rate=int(self.default_speakers["defaultSampleRate"]),
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.default_speakers["index"],
            stream_callback=self._fill_buffer,
        )

    def initialize_speakers(self) -> None:
        """Initialize PyAudio and get default speakers."""
        try:
            # Get default WASAPI info
            wasapi_info = self._audio_interface.get_host_api_info_by_type(pyaudiow.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            raise

        # Get default WASAPI speakers
        self.default_speakers = self._audio_interface.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not self.default_speakers["isLoopbackDevice"]:
            for loopback in self._audio_interface.get_loopback_device_info_generator():
                if self.default_speakers["name"] in loopback["name"]:
                    self.default_speakers = loopback
                    break
            else:
                print(
                    "Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.")
                raise RuntimeError("No loopback device found")
        print(f"Recording from: ({self.default_speakers['index']}){self.default_speakers['name']}")

    def _fill_buffer(
            self,
            in_data: object,
            *args: object,
            **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Converts stereo to mono if needed.

        Args:
            in_data: The audio data as a bytes object.
            args: Additional arguments.
            kwargs: Additional arguments.

        Returns:
            Tuple of (None, paContinue)
        """
        if self.default_speakers["maxInputChannels"] > 1:
            in_data = self.convert_to_mono(in_data)
        self._buff.put(in_data)
        return None, pyaudiow.paContinue

    def convert_to_mono(self, audio_data):
        """Convert stereo audio to mono.

        Args:
            audio_data: The stereo audio data bytes.

        Returns:
            Mono audio data bytes.
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Reshape based on number of channels
        channels = self.default_speakers["maxInputChannels"]
        audio_array = audio_array.reshape((-1, channels))

        # Convert to mono by averaging channels
        mono_array = audio_array.mean(axis=1).astype(np.int16)

        return mono_array.tobytes()
