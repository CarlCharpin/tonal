import sys
import os
# Add the script's directory to the Python path to ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from audio_processing import analyze_audio, analyze_formants
from scipy.io.wavfile import write as write_wav

# Word list remains the same
vietnamese_words = [
    "ma (ghost)", "mà (but)", "má (mother)", "mả (tomb)", "mã (horse)",
    "mạ (rice seedling)", "ba (three)", "cà (eggplant)", "cá (fish)", "cả (all)"
]

# --- Helper function for robust outlier removal ---
def remove_outliers_iqr(pitch_data):
    """
    Removes outliers from pitch data using the IQR method.
    Any data point outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR is considered an outlier.
    """
    # Can't compute IQR on less than 4 points.
    if pitch_data is None or len(pitch_data[~np.isnan(pitch_data)]) < 4:
        return pitch_data

    q1 = np.nanpercentile(pitch_data, 25)
    q3 = np.nanpercentile(pitch_data, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a copy as a float array to allow for NaN values
    cleaned_pitch = pitch_data.copy().astype(float)
    # Replace outliers with NaN so they are not plotted
    cleaned_pitch[(cleaned_pitch < lower_bound) | (cleaned_pitch > upper_bound)] = np.nan
    return cleaned_pitch

# --- Enhanced Plotting Function ---
def create_pitch_plot(history):
    """
    Generates a plot from a list of pitch and intensity contours.
    This version uses robust outlier detection and a corrected plot order.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title("Pitch Contour")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 1. First, process all data to remove outliers and find the correct Y-axis limits.
    all_cleaned_pitches = []
    processed_history = []
    for times, pitch, norm_intensity in history:
        if pitch is not None:
            cleaned_pitch = remove_outliers_iqr(pitch)
            all_cleaned_pitches.extend(cleaned_pitch[~np.isnan(cleaned_pitch)])
            processed_history.append((times, cleaned_pitch, norm_intensity))
        else:
            processed_history.append((times, pitch, norm_intensity)) # Keep None entries

    # 2. Set the Y-axis limits based on the cleaned data.
    if all_cleaned_pitches:
        min_pitch = np.nanmin(all_cleaned_pitches)
        max_pitch = np.nanmax(all_cleaned_pitches)
        # Add some padding to the limits
        y_bottom = min_pitch * 0.9
        y_top = max_pitch * 1.1
        ax.set_ylim(bottom=max(0, y_bottom), top=y_top)

    # 3. Now, draw the amplitude and pitch plots.
    for i, (times, cleaned_pitch, norm_intensity) in enumerate(processed_history):
        if times is not None and cleaned_pitch is not None and norm_intensity is not None:
            # --- Amplitude as background fill (now correctly scaled) ---
            y_fill_max = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.3 + ax.get_ylim()[0]
            y_fill = norm_intensity * y_fill_max

            alpha = 1.0 - (len(history) - 1 - i) * 0.4
            ax.fill_between(times, y_fill, 0, color='gray', alpha=max(0.05, alpha/4), interpolate=True, zorder=1)

            # --- Visualize Pauses in Pitch Plot ---
            plot_pitch = cleaned_pitch.copy()
            plot_pitch[norm_intensity < 0.15] = np.nan # Create breaks for quiet parts

            # Plot the pitch contour
            ax.plot(times, plot_pitch, label=f"Attempt {i+1}", alpha=max(0.2, alpha), marker='.', zorder=10+i)

    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

def create_vowel_plot(formant_history):
    """
    Generates a plot of the F1/F2 vowel trajectory on a reference chart.
    """
    fig, ax = plt.subplots()
    ax.set_title("Vowel Formant Trajectory")

    # Load and display the vowel chart image as the background
    try:
        img = plt.imread("vowel_chart.png")
        # The extent parameter defines the data coordinates for the image corners [left, right, bottom, top].
        # The inverted axes will orient the image correctly.
        ax.imshow(img, aspect='auto', extent=[800, 2500, 200, 900], zorder=0)
    except FileNotFoundError:
        print("vowel_chart.png not found. Plotting on a blank background.")

    # Plot each formant trajectory attempt
    for i, (f1, f2) in enumerate(formant_history):
        if f1 is not None and f2 is not None and len(f1) > 0:
            alpha = 1.0 - (len(formant_history) - 1 - i) * 0.4
            # Plot F2 vs F1, as is standard
            ax.plot(f2, f1, alpha=max(0.2, alpha), marker='o', linestyle='-', markersize=4, label=f"Attempt {i+1}")
            # Add a start point marker
            ax.plot(f2[0], f1[0], marker='>', color='green', markersize=8, alpha=alpha)

    # Set axis limits to create an inverted chart (origin at top-right)
    ax.set_xlim(2500, 800)
    ax.set_ylim(900, 200)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.legend()
    plt.tight_layout()
    return fig

# --- Main Analysis Function for Gradio ---
def analyze_pronunciation(word, audio_input, pitch_history, formant_history):
    """
    Processes audio for both pitch and formants, returning both plots.
    """
    if audio_input is None:
        # Return empty plots and unchanged history if no audio is provided
        return create_pitch_plot([]), create_vowel_plot([]), pitch_history, formant_history

    sample_rate, audio_data = audio_input
    temp_audio_file = "temp_recording.wav"
    write_wav(temp_audio_file, sample_rate, audio_data)

    # --- Pitch Analysis ---
    times, pitch_values, intensity_values = analyze_audio(temp_audio_file)
    if pitch_history is None:
        pitch_history = []
    pitch_history.append((times, pitch_values, intensity_values))
    if len(pitch_history) > 3:
        pitch_history.pop(0)
    pitch_plot_fig = create_pitch_plot(pitch_history)

    # --- Vowel Formant Analysis ---
    f1, f2 = analyze_formants(temp_audio_file)
    if formant_history is None:
        formant_history = []
    formant_history.append((f1, f2))
    if len(formant_history) > 3:
        formant_history.pop(0)
    vowel_plot_fig = create_vowel_plot(formant_history)

    # Return both plots and updated histories
    return pitch_plot_fig, vowel_plot_fig, pitch_history, formant_history

# --- Gradio Interface Definition ---
def main():
    """Defines and launches the Gradio app."""
    with gr.Blocks() as iface:
        gr.Markdown("# Vietnamese Tone & Vowel Visualizer")
        gr.Markdown("Select a word, record yourself, and analyze your pronunciation.")

        with gr.Row():
            with gr.Column(scale=1):
                word_selection = gr.Dropdown(vietnamese_words, label="Select a Word", value=vietnamese_words[0])
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Your Pronunciation")

                with gr.Row():
                    analyze_btn = gr.Button("Analyze Pronunciation")

                with gr.Row():
                    try_again_btn = gr.Button("Try Again")
                    clear_btn = gr.Button("Clear All")

            with gr.Column(scale=2):
                pitch_plot = gr.Plot(label="Pitch Contour")
                vowel_plot = gr.Plot(label="Vowel Formant Trajectory")

        # States for history
        pitch_history_state = gr.State([])
        formant_history_state = gr.State([])

        # Button Clicks
        analyze_btn.click(
            fn=analyze_pronunciation,
            inputs=[word_selection, audio_input, pitch_history_state, formant_history_state],
            outputs=[pitch_plot, vowel_plot, pitch_history_state, formant_history_state]
        )

        def try_again(p_history, f_history):
            """Removes the last attempt from both histories and plots."""
            if p_history: p_history.pop()
            if f_history: f_history.pop()
            # Return empty audio input, updated plots, and updated histories
            return None, create_pitch_plot(p_history), create_vowel_plot(f_history), p_history, f_history

        try_again_btn.click(
            fn=try_again,
            inputs=[pitch_history_state, formant_history_state],
            outputs=[audio_input, pitch_plot, vowel_plot, pitch_history_state, formant_history_state]
        )

        def clear_all():
            """Clears all history and plots."""
            # Return empty audio, empty plots, and empty histories
            return None, create_pitch_plot([]), create_vowel_plot([]), [], []

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[audio_input, pitch_plot, vowel_plot, pitch_history_state, formant_history_state]
        )

    print("Launching Gradio app...")
    iface.launch()

if __name__ == "__main__":
    main()