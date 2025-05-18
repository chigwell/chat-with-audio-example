[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/eugene-evstafev-716669181/)

# Chat With Audio Example

This Python project demonstrates how to integrate audio processing with a conversational retrieval chain using various libraries including LangChain, pydub, and SpeechRecognition. The application converts spoken language to text, processes it, and utilizes a conversational AI model to generate responses.

## Description

The script `main.py` performs the following functions:
- Splits an audio file into segments.
- Recognizes speech from these audio segments and converts it into text.
- Uses this text to interact with a conversational model, processing the conversation context and generating responses dynamically.

## Installation

### Requirements

This project requires Python 3.6+ and pip for managing Python packages. Before running the script, you need to set up a virtual environment and install the required dependencies.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chigwell/chat-with-audio-example
   cd chat-with-audio-example
   ```

2. **Create and activate a virtual environment**:
   - On Unix/MacOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```cmd
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```

### External Dependencies

The script uses the `ChatOllama` model hosted on Ollama. You need to follow these steps to install and use models from Ollama:

- Visit [Ollama Models](https://ollama.com/library/mistral) to find the `mistral` model.
- Follow the installation and usage instructions provided on Ollama's website to configure the model for use with this script.

## Usage

Once the installation is complete, you can run the script with an audio file (WAV format) as an argument:

```bash
python main.py example.wav
```

Make sure to replace `example.wav` with the path to your actual audio file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
