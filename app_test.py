# This script is for a Gradio application to visually determine the optimal time_step
# for vowel formant path extraction and plotting.

import gradio as gr
import plotly.graph_objects as go
import numpy as np
import parselmouth
from scipy.io.wavfile import write as write_wav
from scipy.interpolate import interp1d

# --- Audio Processing Functions ---

def moving_average(data, window_size=3):
    """Applies a simple moving average to smooth data."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def analyze_formants(audio_filepath, time_step=0.01, num_points=20):
    """
    Analyzes an audio file to extract, smooth, and time-normalize its F1 and F2 formant trajectories.

    Args:
        audio_filepath (str): The path to the audio file.
        time_step (float): The time step for formant analysis.
        num_points (int): The number of points to normalize the trajectory to.

    Returns:
        A tuple of (f1_normalized, f2_normalized) for the voiced sections.
        Returns (None, None) if analysis fails.
    """
    try:
        snd = parselmouth.Sound(audio_filepath)
        # Use a pitch object to find voiced parts of the sound
        pitch = snd.to_pitch(time_step=time_step)

        # Extract formants at the specified time_step
        formant = snd.to_formant_burg(time_step=time_step, max_number_of_formants=5, maximum_formant=5500.0)

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

        if not f1_raw or not f2_raw or len(f1_raw) < 4:
            return (None, None)

        # 1. Smoothing (Pre-Processing)
        f1_smoothed = moving_average(np.array(f1_raw))
        f2_smoothed = moving_average(np.array(f2_raw))

        if len(f1_smoothed) < 4: # Need enough points for interpolation
            return (None, None)

        # 2. Time Normalization (Uniform Sampling)
        # Create an interpolation function for the smoothed data
        duration = len(f1_smoothed)
        x_original = np.linspace(0, 1, duration)
        # Use cubic interpolation; requires at least 4 points
        f1_interp = interp1d(x_original, f1_smoothed, kind='cubic', fill_value="extrapolate")
        f2_interp = interp1d(x_original, f2_smoothed, kind='cubic', fill_value="extrapolate")

        # Resample to a fixed number of points
        x_normalized = np.linspace(0, 1, num_points)
        f1_normalized = f1_interp(x_normalized)
        f2_normalized = f2_interp(x_normalized)

        return (f1_normalized, f2_normalized)

    except Exception as e:
        print(f"Error analyzing formants for time_step {time_step}: {e}")
        return (None, None)

# --- Plotting Function ---

def create_vowel_plot(formant_data, time_step):
    """
    Generates a plot of the F1/F2 vowel trajectory for a given time_step.
    """
    f1, f2 = formant_data
    fig = go.Figure()

    # Standard IPA vowel formant data
    vowel_data = {
        'i': {'f1': 270, 'f2': 2290}, 'y': {'f1': 270, 'f2': 2000},
        'ɪ': {'f1': 390, 'f2': 1990}, 'ʏ': {'f1': 390, 'f2': 1800},
        'e': {'f1': 450, 'f2': 2200}, 'ø': {'f1': 450, 'f2': 1850},
        'ɛ': {'f1': 530, 'f2': 1840}, 'œ': {'f1': 530, 'f2': 1600},
        'æ': {'f1': 660, 'f2': 1720},
        'a': {'f1': 700, 'f2': 1400}, 'ɶ': {'f1': 700, 'f2': 1500},
        'ɑ': {'f1': 730, 'f2': 1090}, 'ɒ': {'f1': 730, 'f2': 900},
        'ʌ': {'f1': 640, 'f2': 1220}, 'ɔ': {'f1': 570, 'f2': 840},
        'ɤ': {'f1': 450, 'f2': 1000}, 'o': {'f1': 450, 'f2': 800},
        'ɯ': {'f1': 300, 'f2': 900}, 'u': {'f1': 300, 'f2': 870},
        'ə': {'f1': 500, 'f2': 1500}
    }

    # Draw the IPA vowel points and their labels
    f1_vals = [v['f1'] for v in vowel_data.values()]
    f2_vals = [v['f2'] for v in vowel_data.values()]
    symbols = list(vowel_data.keys())

    fig.add_trace(go.Scatter(
        x=f2_vals, y=f1_vals, mode='text', text=symbols,
        textfont=dict(size=16, color='black'), showlegend=False
    ))

    # Draw the vowel chart trapezoid
    fig.add_shape(type="path",
        path=" M 2290,270 L 1720,660 L 1090,730 L 840,570 L 800,450 L 900,300 L 2290,270 Z",
        line=dict(color="lightgrey", width=2), layer="below"
    )

    # Plot the formant trajectory
    if f1 is not None and f2 is not None and len(f1) > 0:
        fig.add_trace(go.Scatter(
            x=f2, y=f1, mode='lines+markers', name='Vowel Path',
            line=dict(width=3, color='blue'), marker=dict(size=8),
            hovertemplate='F2: %{x:.0f}Hz<br>F1: %{y:.0f}Hz<extra></extra>'
        ))
        # Start and End markers
        fig.add_trace(go.Scatter(
            x=[f2[0]], y=[f1[0]], mode='markers',
            marker=dict(symbol='triangle-right', color='green', size=12),
            name='Start', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[f2[-1]], y=[f1[-1]], mode='markers',
            marker=dict(symbol='circle', color='red', size=12),
            name='End', showlegend=False
        ))

    # Invert axes and set layout
    fig.update_layout(
        title=f"Vowel Path (Time Step: {time_step:.2f}s)",
        xaxis_title="F2 (Hz)",
        yaxis_title="F1 (Hz)",
        xaxis=dict(range=[2500, 750]),
        yaxis=dict(range=[900, 200]),
        template="plotly_white",
        width=600,
        height=600,
        showlegend=False
    )

    return fig

# --- Main Analysis Function for Gradio ---

def analyze_all_time_steps(audio_input):
    """
    Processes audio and generates a vowel plot for each time_step value.
    """
    if audio_input is None:
        # Return 10 empty plots if no audio is provided
        return [create_vowel_plot((None, None), ts) for ts in np.arange(0.01, 0.11, 0.01)]

    sample_rate, audio_data = audio_input
    temp_audio_file = "temp_app_test_audio.wav"
    write_wav(temp_audio_file, sample_rate, audio_data)

    time_steps = np.arange(0.01, 0.11, 0.01)
    output_plots = []

    for ts in time_steps:
        formant_data = analyze_formants(temp_audio_file, time_step=ts)
        fig = create_vowel_plot(formant_data, ts)
        output_plots.append(fig)

    # The function needs to return a list of plots that matches the number of gr.Plot outputs
    return output_plots[0], output_plots[1], output_plots[2], output_plots[3], output_plots[4], output_plots[5], output_plots[6], output_plots[7], output_plots[8], output_plots[9]

# --- Gradio Interface Definition ---
def main():
    """Defines and launches the Gradio app."""
    with gr.Blocks() as iface:
        gr.Markdown("# Vowel Formant Path - Time Step Experiment")
        gr.Markdown("Upload an audio file or record yourself to see how the `time_step` parameter affects the vowel formant trajectory.")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record or Upload Audio")
                analyze_btn = gr.Button("Analyze Audio", variant="primary")

        gr.Markdown("---")
        gr.Markdown("### Vowel Trajectory Plots per Time Step")

        # Create 10 plot components, stacked vertically
        plot_outputs = []
        with gr.Blocks():
            for i in range(10):
                 # Gradio plots in two columns automatically, let's force one
                with gr.Row():
                    plot = gr.Plot()
                    plot_outputs.append(plot)

        # When the button is clicked, run the analysis and update all 10 plots
        analyze_btn.click(
            fn=analyze_all_time_steps,
            inputs=[audio_input],
            outputs=plot_outputs
        )

    print("Launching Gradio app...")
    iface.launch()

if __name__ == "__main__":
    main()