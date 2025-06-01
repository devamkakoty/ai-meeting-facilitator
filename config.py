"""
Configuration Module for the Copilot.

Central place for all configuration settings.
"""
import os
from typing import Dict, Any, Optional

# Audio Recording Parameters
STREAMING_LIMIT = 240000  # 4 minutes

# Microphone settings
MIC_AUDIO_SAMPLE_RATE = 16000
MIC_AUDIO_CHUNK_SIZE = int(MIC_AUDIO_SAMPLE_RATE / 10)  # 100ms chunks

# System audio settings
SYS_SAMPLE_RATE = 48000
SYS_AUDIO_CHUNK_SIZE = 50  # Small chunk size for system audio

# Speech recognition settings
DEFAULT_LANGUAGE_CODE = "en-US"
MAX_ALTERNATIVES = 1

# Display colors
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

# Google Cloud service account credentials
DEFAULT_CREDENTIALS_PATH = "googleServiceAccountCredentials.json"
PINECONE_API_KEY= "pcsk_3DxeCk_EBLSPYzsFox6yaa6ngWSQ7QLZGkna9nt45WBzsVfZq6wKdhm3uPRNBA4xpjthLT"
GOOGLE_API_KEY = "AIzaSyCTtYKvAPOFE5S6qIFEpVAAy3eU-M1TCMc"

class Config:
    """Configuration manager for the application."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_file: Optional path to a configuration file.
        """
        self.config_data = {
            "streaming_limit": STREAMING_LIMIT,
            "mic_audio_sample_rate": MIC_AUDIO_SAMPLE_RATE,
            "mic_audio_chunk_size": MIC_AUDIO_CHUNK_SIZE,
            "sys_audio_chunk_size": SYS_AUDIO_CHUNK_SIZE,
            "language_code": DEFAULT_LANGUAGE_CODE,
            "max_alternatives": MAX_ALTERNATIVES,
            "credentials_path": DEFAULT_CREDENTIALS_PATH,
        }

        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from a file.

        Args:
            config_file: Path to the configuration file.
        """
        # TODO: Implement file-based configuration loading
        # This could use JSON, YAML, or other formats based on needs
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: The configuration key.
            default: Default value if key is not found.

        Returns:
            The configuration value.
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: The configuration key.
            value: The value to set.
        """
        self.config_data[key] = value

    def get_credentials_path(self) -> str:
        """Get the path to Google Cloud credentials.

        Returns:
            String path to credentials file.
        """
        return self.get("credentials_path", DEFAULT_CREDENTIALS_PATH)

    def get_mic_audio_config(self) -> Dict[str, Any]:
        """Get microphone configuration.

        Returns:
            Dictionary with microphone configuration.
        """
        return {
            "sample_rate": self.get("mic_sample_rate", MIC_AUDIO_SAMPLE_RATE),
            "chunk_size": self.get("mic_chunk_size", MIC_AUDIO_CHUNK_SIZE),
        }

    def get_sys_audio_config(self) -> Dict[str, Any]:
        """Get system audio configuration.

        Returns:
            Dictionary with system audio configuration.
        """
        return {
            "chunk_size": self.get("sys_chunk_size", SYS_AUDIO_CHUNK_SIZE),
        }
