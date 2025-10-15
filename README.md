# Vietnamese Tone Visualizer

This is a simple web application designed to help learners of the Vietnamese language visualize their pitch when pronouncing words. By seeing a graph of their tone, users can better understand and practice the six tones of Vietnamese.

This application is built with Python using the Gradio library for the user interface.

## Setup and Installation

To run this application, you will need Python 3 installed on your system.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install the required dependencies:**
    The necessary Python libraries are listed in the `requirements.txt` file. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If a `requirements.txt` file is not available, you can install the packages manually:*
    ```bash
    pip install gradio praat-parselmouth matplotlib pandas scipy
    ```

## How to Run the Application

Once the dependencies are installed, you can start the application by running the `app.py` script:

```bash
python app.py
```

This will start a local web server. You will see a URL in your terminal (usually `http://127.0.0.1:7860`). Open this URL in your web browser to use the application.

## How to Use

1.  Select a Vietnamese word from the dropdown menu.
2.  Click the "Record Your Pronunciation" button and speak the word into your microphone.
3.  Click "Analyze" to see the pitch contour of your voice.
4.  Your last three attempts will be shown on the graph for comparison. Use the "Clear History" button to start fresh.