# This module will contain the core logic for processing audio
# and extracting pitch contours. It is designed to be reusable.

import parselmouth
import numpy as np

def extract_pitch(audio_filepath):
    """
    Analyzes an audio file to extract its pitch (F0) contour using Parselmouth.

    Args:
        audio_filepath (str): The path to the audio file (e.g., a .wav file).

    Returns:
        A tuple of (times, pitch_values) where pitch_values are in Hz.
        Returns (None, None) if pitch cannot be extracted.
    """
    try:
        # Load the sound file
        snd = parselmouth.Sound(audio_filepath)

        # Extract the pitch
        # We use a time step of 0.01 seconds (10 ms)
        # Pitch floor and ceiling are set to typical human voice ranges
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)

        # Get the pitch values and the corresponding times
        pitch_values = pitch.selected_array['frequency']

        # In Parselmouth, a frequency of 0 indicates that the sound is unvoiced
        # We will replace 0s with numpy.nan to avoid plotting them
        pitch_values[pitch_values == 0] = np.nan

        times = pitch.xs()

        return (times, pitch_values)

    except Exception as e:
        # This can happen if the file is not a valid audio file
        # or if there's an issue with the analysis.
        print(f"Error extracting pitch: {e}")
        return (None, None)