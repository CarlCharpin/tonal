import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from audio_processing import extract_pitch

# 1. Word List: 10 Vietnamese words with various tones.
vietnamese_words = [
    "ma (ghost)",      # Ngang
    "mà (but)",        # Huyền
    "má (mother)",     # Sắc
    "mả (tomb)",       # Hỏi
    "mã (horse)",      # Ngã
    "mạ (rice seedling)",# Nặng
    "ba (three)",      # Ngang
    "cà (eggplant)",   # Huyền
    "cá (fish)",       # Sắc
    "cả (all)"         # Hỏi
]

# 2. Plotting Function
def create_pitch_plot(history):
    """Generates a plot from a list of pitch contours."""
    fig, ax = plt.subplots()

    # Plot each attempt in the history
    for i, (times, pitch_values) in enumerate(history):
        if times is not None and pitch_values is not None:
            # The most recent attempt is solid, older ones are faded.
            alpha = 1.0 - (len(history) - 1 - i) * 0.4
            ax.plot(times, pitch_values, label=f"Attempt {i+1}", alpha=max(0.2, alpha), marker='.')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.set_title("Pitch Contour")
    ax.legend()
    ax.grid(True)
    # Ensure y-axis starts from a reasonable place if data is present
    if history and any(h[1] is not None for h in history):
        all_pitches = np.concatenate([h[1][~np.isnan(h[1])] for h in history if h[1] is not None])
        if len(all_pitches) > 0:
            ax.set_ylim(bottom=max(0, np.min(all_pitches) - 20))

    plt.tight_layout()
    return fig

# 3. Main Analysis Function for Gradio
def analyze_pronunciation(word, audio_input, history_state):
    """
    Processes the user's audio, extracts pitch, and returns a plot.
    """
    if audio_input is None:
        # If no audio is provided, return a blank plot and original history
        return create_pitch_plot([]), history_state

    # The audio_input from Gradio is a tuple (sample_rate, numpy_array)
    # We need to save it to a temporary file to use with parselmouth
    sample_rate, audio_data = audio_input
    temp_audio_file = "temp_recording.wav"

    # The parselmouth library reads from a file path, so we write a temporary wav file.
    # We need to import scipy for this.
    from scipy.io.wavfile import write as write_wav
    write_wav(temp_audio_file, sample_rate, audio_data)

    # Extract pitch from the saved file
    times, pitch_values = extract_pitch(temp_audio_file)

    # Update the history state
    if history_state is None:
        history_state = []

    # Add the new attempt to the history
    history_state.append((times, pitch_values))

    # Keep only the last 3 attempts for comparison
    if len(history_state) > 3:
        history_state.pop(0)

    # Generate the plot and return it with the updated state
    plot = create_pitch_plot(history_state)

    return plot, history_state

# 4. Gradio Interface Definition
def main():
    """Defines and launches the Gradio app."""
    with gr.Blocks() as iface:
        gr.Markdown("# Vietnamese Tone Visualizer")
        gr.Markdown("Select a word, record yourself saying it, and see your pitch contour!")

        with gr.Row():
            with gr.Column(scale=1):
                word_selection = gr.Dropdown(vietnamese_words, label="Select a Word", value=vietnamese_words[0])
                audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record Your Pronunciation")
                submit_btn = gr.Button("Analyze")

            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Pitch Contour")

        # State component to store the history of attempts
        # It's not visible to the user.
        history_state = gr.State([])

        submit_btn.click(
            fn=analyze_pronunciation,
            inputs=[word_selection, audio_input, history_state],
            outputs=[output_plot, history_state]
        )

        # Add a clear button to reset the history
        def clear_history():
            return create_pitch_plot([]), []

        clear_btn = gr.Button("Clear History")
        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[output_plot, history_state]
        )

    print("Launching Gradio app...")
    iface.launch()

if __name__ == "__main__":
    main()