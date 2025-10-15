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
        intensity_values = intensity.values[0]

        # --- Normalize Intensity to a 0-1 scale ---
        # This makes it a simple visual guide without needing a dB scale.
        min_intensity = np.nanmin(intensity_values)
        max_intensity = np.nanmax(intensity_values)
        if max_intensity > min_intensity:
            normalized_intensity = (intensity_values - min_intensity) / (max_intensity - min_intensity)
        else:
            # Avoid division by zero if intensity is constant
            normalized_intensity = np.zeros_like(intensity_values)

        # --- Time Alignment ---
        times = pitch.xs()

        min_len = min(len(pitch_values), len(normalized_intensity))
        times = times[:min_len]
        pitch_values = pitch_values[:min_len]
        normalized_intensity = normalized_intensity[:min_len]

        return (times, pitch_values, normalized_intensity)

    except Exception as e:
        # This can happen if the file is not a valid audio file
        # or if there's an issue with the analysis.
        print(f"Error analyzing audio: {e}")
        return (None, None, None)