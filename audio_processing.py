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

from scipy.interpolate import interp1d

def moving_average(data, window_size=3):
    """Applies a simple moving average to smooth data."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def analyze_formants(audio_filepath, num_points=20):
    """
    Analyzes an audio file to extract, smooth, and time-normalize its F1 and F2 formant trajectories.

    Args:
        audio_filepath (str): The path to the audio file.
        num_points (int): The number of points to normalize the trajectory to.

    Returns:
        A tuple of (f1_normalized, f2_normalized) for the voiced sections.
        Returns (None, None) if analysis fails.
    """
    try:
        snd = parselmouth.Sound(audio_filepath)
        pitch = snd.to_pitch()
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)

        voiced_times = pitch.xs()[pitch.selected_array['frequency'] > 0]
        if len(voiced_times) < 4: # Need enough points for analysis
            return (None, None)

        f1_raw = []
        f2_raw = []
        for t in voiced_times:
            f1 = formant.get_value_at_time(formant_number=1, time=t)
            f2 = formant.get_value_at_time(formant_number=2, time=t)
            if not np.isnan(f1) and not np.isnan(f2):
                f1_raw.append(f1)
                f2_raw.append(f2)

        if not f1_raw or not f2_raw:
            return (None, None)

        # 1. Smoothing (Pre-Processing)
        f1_smoothed = moving_average(np.array(f1_raw))
        f2_smoothed = moving_average(np.array(f2_raw))

        # 2. Time Normalization (Uniform Sampling)
        # Create an interpolation function for the smoothed data
        duration = len(f1_smoothed)
        x_original = np.linspace(0, 1, duration)
        f1_interp = interp1d(x_original, f1_smoothed, kind='cubic')
        f2_interp = interp1d(x_original, f2_smoothed, kind='cubic')

        # Resample to a fixed number of points
        x_normalized = np.linspace(0, 1, num_points)
        f1_normalized = f1_interp(x_normalized)
        f2_normalized = f2_interp(x_normalized)

        return (f1_normalized, f2_normalized)

    except Exception as e:
        print(f"Error analyzing formants: {e}")
        return (None, None)