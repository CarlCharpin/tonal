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

def analyze_formants(audio_filepath):
    """
    Analyzes an audio file to extract its F1 and F2 formant trajectories.

    Args:
        audio_filepath (str): The path to the audio file.

    Returns:
        A tuple of (f1_values, f2_values) for the voiced sections.
        Returns (None, None) if analysis fails.
    """
    try:
        snd = parselmouth.Sound(audio_filepath)
        # To get formants only for voiced sections, we first need the pitch
        pitch = snd.to_pitch()

        # This is the formant object. We need to specify max formants and max frequency.
        # 5500 Hz is standard for female speakers, 5000 Hz for male. We'll use 5500.
        formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)

        # Get the times from the pitch object where voicing is detected
        voiced_times = pitch.xs()[pitch.selected_array['frequency'] > 0]

        f1_values = []
        f2_values = []

        for t in voiced_times:
            f1 = formant.get_value_at_time(formant_number=1, time=t)
            f2 = formant.get_value_at_time(formant_number=2, time=t)
            # Only add the formants if they are not NaN
            if not np.isnan(f1) and not np.isnan(f2):
                f1_values.append(f1)
                f2_values.append(f2)

        return (np.array(f1_values), np.array(f2_values))

    except Exception as e:
        print(f"Error analyzing formants: {e}")
        return (None, None)