# This module will contain the core logic for processing audio
# and extracting pitch and intensity contours. It is designed to be reusable.

import parselmouth
import numpy as np

def analyze_audio(audio_filepath):
    """
    Analyzes an audio file to extract its pitch (F0) and intensity contours.

    Args:
        audio_filepath (str): The path to the audio file (e.g., a .wav file).

    Returns:
        A tuple of (times, pitch_values, intensity_values).
        Returns (None, None, None) if analysis fails.
    """
    try:
        # Load the sound file
        snd = parselmouth.Sound(audio_filepath)

        # --- Pitch Extraction ---
        # We use a time step of 0.01 seconds (10 ms)
        # Pitch floor and ceiling are set to typical human voice ranges
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        pitch_values = pitch.selected_array['frequency']
        # In Parselmouth, a frequency of 0 indicates that the sound is unvoiced.
        # We will replace 0s with numpy.nan to avoid plotting them.
        pitch_values[pitch_values == 0] = np.nan

        # --- Intensity Extraction ---
        intensity = snd.to_intensity(time_step=0.01)
        # We get the intensity values in dB.
        intensity_values = intensity.values[0] # Intensity is a 1D array inside a 2D one.

        # --- Time Alignment ---
        # We'll use the pitch object's time values as the reference.
        times = pitch.xs()

        # It's possible for intensity to have a slightly different number of
        # time steps. We'll truncate the longer array to match the shorter one.
        min_len = min(len(pitch_values), len(intensity_values))
        times = times[:min_len]
        pitch_values = pitch_values[:min_len]
        intensity_values = intensity_values[:min_len]

        return (times, pitch_values, intensity_values)

    except Exception as e:
        # This can happen if the file is not a valid audio file
        # or if there's an issue with the analysis.
        print(f"Error analyzing audio: {e}")
        return (None, None, None)