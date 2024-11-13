# Protocol Creation

This project uses OpenAI's whisper to transcribe an audio file, a clustering method for speaker diarization and Meta's Llama3.2 model to create a summary.

### Get Started

1. Clone this repository. 

```
$ git clone https://github.com/Flosener/protocol-creation.git
```

2. Create a conda environment and install packages from requirements file.

```
$ conda create -n NAME python=3.90.2
$ conda activate NAME
$ pip install -r requirements.txt
```

3. Run the main script using the arguments below.

```
$ python main.py --audio PATH_TO_AUDIO_FILE --protocol PATH_TO_TEXT_FILE --summarize --token YOUR_HF_TOKEN
```

- audio: If you start with an audio file that you want to transcript and diarize, and potentially summarize, input the path to the mp3 file.
- protocol: If you already have a .txt file that you only want to summarize you do not need --audio, but only input the path to the txt file.
- summarize: A boolean flag that indicates whether you want to summarize the text file as well, or just transcribe and diarize speakers.
- token: Requires 'summarize' flag to be true. You have to specify a huggingface token as authorization if you want to use the Llama model.

**Note:** If you want to use summarization you need a hugging face account, generate a token and request access to the Llama3.2 models on HF: https://huggingface.co/meta-llama/Llama-3.2-3B .

### Sources
Transcription code: https://www.youtube.com/watch?v=MVW746z8y_I
Summarization code: https://www.youtube.com/watch?v=fc7cAP5zrOY