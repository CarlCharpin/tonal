import sys
import os
# Add the script's directory to the Python path to ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from audio_processing import analyze_audio
from scipy.io.wavfile import write as write_wav

# Word list remains the same
vietnamese_words = [
    "ma (ghost)", "mà (but)", "má (mother)", "mả (tomb)", "mã (horse)",
    "mạ (rice seedling)", "ba (three)", "cà (eggplant)", "cá (fish)", "cả (all)"
]

# --- Enhanced Plotting Function ---
def create_pitch_plot(history):
    """
    Generates a plot from a list of pitch and intensity contours.
    Handles outliers, visualizes pauses, and shows amplitude.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title("Pitch Contour")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    all_pitches = []

    # Collect all valid pitch data first for scaling
    for _, pitch, _ in history:
        if pitch is not None:
            all_pitches.extend(pitch[~np.isnan(pitch)])

    # --- Pitch and Amplitude Plotting ---
    for i, (times, pitch, norm_intensity) in enumerate(history):
        if times is not None and pitch is not None and norm_intensity is not None:
            # --- Amplitude as background fill ---
            # The normalized intensity (0 to 1) is scaled to the bottom 30% of the pitch range
            # to serve as a visual guide without cluttering the main plot.
            y_fill_max = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.3 + ax.get_ylim()[0]
            y_fill = norm_intensity * y_fill_max

            alpha = 1.0 - (len(history) - 1 - i) * 0.4
            ax.fill_between(times, y_fill, 0, color='gray', alpha=max(0.05, alpha/4), interpolate=True)

            # --- Visualize Pauses in Pitch Plot ---
            # Use a threshold on the normalized intensity to detect silence
            plot_pitch = pitch.copy()
            plot_pitch[norm_intensity < 0.15] = np.nan # Create breaks in the line for quiet parts

            # The most recent attempt is solid, older ones are faded.
            alpha = 1.0 - (len(history) - 1 - i) * 0.4
            ax.plot(times, plot_pitch, label=f"Attempt {i+1}", alpha=max(0.2, alpha), marker='.', zorder=10+i)

    # --- Handle Outliers for Y-axis scaling ---
    if all_pitches:
        # Use 98th percentile to avoid outliers skewing the graph
        upper_bound = np.nanpercentile(all_pitches, 98) * 1.1
        lower_bound = np.nanpercentile(all_pitches, 2) * 0.9
        ax.set_ylim(bottom=max(0, lower_bound), top=upper_bound)

    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

# --- Main Analysis Function for Gradio ---
def analyze_pronunciation(word, audio_input, history_state):
    """
    Processes user's audio, calls the analysis module, and returns a plot.
    """
    if audio_input is None:
        return create_pitch_plot([]), history_state

    sample_rate, audio_data = audio_input
    temp_audio_file = "temp_recording.wav"
    write_wav(temp_audio_file, sample_rate, audio_data)

    # Call the new, upgraded analysis function
    times, pitch_values, intensity_values = analyze_audio(temp_audio_file)

    if history_state is None:
        history_state = []

    # Add new attempt (now a 3-tuple) to history
    history_state.append((times, pitch_values, intensity_values))

    if len(history_state) > 3:
        history_state.pop(0)

    plot = create_pitch_plot(history_state)

    return plot, history_state

# --- Gradio Interface Definition ---
def main():
    """Defines and launches the Gradio app."""
    with gr.Blocks() as iface:
        gr.Markdown("# Vietnamese Tone Visualizer")
        gr.Markdown("Select a word, record yourself saying it, and see your pitch contour!")

        with gr.Row():
            with gr.Column(scale=1):
                word_selection = gr.Dropdown(vietnamese_words, label="Select a Word", value=vietnamese_words[0])
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Your Pronunciation")

                with gr.Row():
                    submit_btn = gr.Button("Analyze")
                    try_again_btn = gr.Button("Try Again") # New Button

                clear_btn = gr.Button("Clear All History")

            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Pitch Contour")

        history_state = gr.State([])

        submit_btn.click(
            fn=analyze_pronunciation,
            inputs=[word_selection, audio_input, history_state],
            outputs=[output_plot, history_state]
        )

        # --- "Try Again" Functionality ---
        def try_again(history):
            """Removes the last attempt from the history and plot."""
            if history:
                history.pop()
            # Return a cleared audio input widget and the updated plot/history
            return None, create_pitch_plot(history), history

        try_again_btn.click(
            fn=try_again,
            inputs=[history_state],
            outputs=[audio_input, output_plot, history_state]
        )

        # --- "Clear History" Functionality ---
        def clear_history():
            """Clears all attempts and the plot."""
            return create_pitch_plot([]), []

        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[output_plot, history_state]
        )

    print("Launching Gradio app...")
    iface.launch()

if __name__ == "__main__":
    main()