import sys
import os
# Add the script's directory to the Python path to ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    Generates an interactive plot from a list of pitch and intensity contours using Plotly.
    """
    fig = go.Figure()

    all_cleaned_pitches = []
    processed_history = []

    # 1. First, process all data to remove outliers and find the correct Y-axis limits.
    for times, pitch, norm_intensity in history:
        if pitch is not None:
            cleaned_pitch = remove_outliers_iqr(pitch)
            # Add to list for y-axis calculation, ignoring NaNs
            all_cleaned_pitches.extend(cleaned_pitch[~np.isnan(cleaned_pitch)])
            processed_history.append((times, cleaned_pitch, norm_intensity))
        else:
            processed_history.append((times, pitch, norm_intensity)) # Keep None entries

    # 2. Set the Y-axis range based on the cleaned data.
    y_range = None
    if all_cleaned_pitches:
        min_pitch = np.nanmin(all_cleaned_pitches)
        max_pitch = np.nanmax(all_cleaned_pitches)
        y_bottom = max(0, min_pitch * 0.9)
        y_top = max_pitch * 1.1
        y_range = [y_bottom, y_top]

    # 3. Now, draw the amplitude and pitch plots.
    for i, (times, cleaned_pitch, norm_intensity) in enumerate(processed_history):
        if times is not None and cleaned_pitch is not None and norm_intensity is not None:

            # --- Visualize Pauses in Pitch Plot ---
            plot_pitch = cleaned_pitch.copy()
            plot_pitch[norm_intensity < 0.15] = np.nan # Create breaks for quiet parts

            # --- Amplitude as background fill ---
            # Plotly doesn't have a direct equivalent of zorder, so we plot intensity first.
            # We scale intensity to be 30% of the pitch range for visual guidance.
            if y_range:
                y_fill = norm_intensity * (y_range[1] - y_range[0]) * 0.3 + y_range[0]
                fig.add_trace(go.Scatter(
                    x=times, y=y_fill,
                    fill='tozeroy',
                    mode='none',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    name=f'Intensity {i+1}',
                    showlegend=False,
                ))

            # --- Plot the pitch contour ---
            opacity = 1.0 - (len(history) - 1 - i) * 0.3
            fig.add_trace(go.Scatter(
                x=times,
                y=plot_pitch,
                mode='lines+markers',
                name=f'Attempt {i+1}',
                opacity=max(0.2, opacity),
                marker=dict(size=4),
                hovertemplate='Time: %{x:.2f}s<br>Pitch: %{y:.2f}Hz<extra></extra>'
            ))

    fig.update_layout(
        title="Pitch Contour",
        xaxis_title="Time (s)",
        yaxis_title="Pitch (Hz)",
        yaxis_range=y_range,
        legend_title="Attempts",
        template="plotly_white"
    )

    return fig

def create_vowel_plot(formant_history):
    """
    Generates an interactive plot of the F1/F2 vowel trajectory on a reference chart using Plotly.
    """
    fig = go.Figure()

    # Plot each formant trajectory attempt
    for i, (f1, f2) in enumerate(formant_history):
        if f1 is not None and f2 is not None and len(f1) > 0:
            opacity = 1.0 - (len(formant_history) - 1 - i) * 0.3
            # Plot F2 vs F1, as is standard for vowel charts
            fig.add_trace(go.Scatter(
                x=f2,
                y=f1,
                mode='lines+markers',
                name=f'Attempt {i+1}',
                opacity=max(0.2, opacity),
                marker=dict(size=6),
                hovertemplate='F2: %{x:.0f}Hz<br>F1: %{y:.0f}Hz<extra></extra>'
            ))
            # Add a start point marker
            fig.add_trace(go.Scatter(
                x=[f2[0]],
                y=[f1[0]],
                mode='markers',
                marker=dict(symbol='triangle-right', color='green', size=12),
                name='Start',
                showlegend=False,
                hovertemplate='Start<extra></extra>'
            ))

    # Load and display the vowel chart image as the background
    try:
        from PIL import Image
        img = Image.open("vowel_chart.png")
        fig.add_layout_image(
            dict(
                source=img,
                xref="x",
                yref="y",
                x=800,  # F2 start
                y=200,  # F1 start
                sizex=1700, # F2 range (2500-800)
                sizey=700, # F1 range (900-200)
                sizing="stretch",
                opacity=0.5,
                layer="below")
        )
    except FileNotFoundError:
        print("vowel_chart.png not found. Plotting on a blank background.")

    # Invert axes to match standard phonetic charts (origin at top-right)
    fig.update_layout(
        title="Vowel Formant Trajectory",
        xaxis_title="F2 (Hz)",
        yaxis_title="F1 (Hz)",
        xaxis=dict(range=[2500, 800]),  # Inverted F2 axis
        yaxis=dict(range=[900, 200]),  # Inverted F1 axis
        legend_title="Attempts",
        template="plotly_white"
    )

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
                audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Audio")

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