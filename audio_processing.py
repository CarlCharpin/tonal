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

def analyze_formants(audio_filepath, time_window_ms=50):
    """
    Analyzes the most stable part of a vowel in an audio file to extract its F1 and F2 formants.
    This method finds the point of maximum intensity within voiced segments and analyzes a small
    window around that point.

    Args:
        audio_filepath (str): The path to the audio file.
        time_window_ms (int): The duration of the analysis window in milliseconds.

    Returns:
        A tuple of (f1_values, f2_values) for the stable vowel segment.
        Returns (None, None) if analysis fails or no vowel is found.
    """
    try:
        snd = parselmouth.Sound(audio_filepath)

        # 1. Get intensity and pitch to identify voiced segments
        intensity = snd.to_intensity()
        pitch = snd.to_pitch()

        voiced_times = pitch.xs()[pitch.selected_array['frequency'] > 0]
        if len(voiced_times) == 0:
            print("No voiced segments found.")
            return (None, None)

        # 2. Find the time of maximum intensity within the voiced segments
        max_intensity = 0
        time_of_max_intensity = 0
        for t in voiced_times:
            current_intensity = intensity.get_value(t)
            if current_intensity > max_intensity:
                max_intensity = current_intensity
                time_of_max_intensity = t

        if time_of_max_intensity == 0:
            print("Could not determine the point of maximum intensity.")
            return (None, None)

        # 3. Define the analysis window around the point of maximum intensity
        window_duration_s = time_window_ms / 1000.0
        start_time = max(0, time_of_max_intensity - (window_duration_s / 2))
        end_time = min(snd.duration, time_of_max_intensity + (window_duration_s / 2))

        # Extract the sound segment for analysis
        vowel_segment = snd.extract_part(from_time=start_time, to_time=end_time)

        # 4. Analyze formants on the extracted vowel segment
        formant = vowel_segment.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500.0)

        num_time_steps = formant.get_number_of_frames()
        f1_values = []
        f2_values = []

        for i in range(num_time_steps):
            time = formant.get_time_from_frame_number(i+1)
            f1 = formant.get_value_at_time(formant_number=1, time=time)
            f2 = formant.get_value_at_time(formant_number=2, time=time)

            if not np.isnan(f1) and not np.isnan(f2):
                f1_values.append(f1)
                f2_values.append(f2)

        return (np.array(f1_values), np.array(f2_values))

    except Exception as e:
        print(f"Error analyzing formants: {e}")
        return (None, None)