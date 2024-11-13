import argparse
import sys
from transcribe import Transcriber
from summarize import Summarizer
from huggingface_hub import login
print('LOADED PACKAGES.')

# For Testing
import warnings
warnings.filterwarnings("ignore")

def main(audio=None, protocol=None, summarize=False, token=None):

  if audio:
    transcriber = Transcriber(num_speakers=4, language='English', model_size='medium')
    result, duration = transcriber.transcribe(audio)
    segments = result['segments']
    embeddings = transcriber.generate_embeddings(segments, duration, audio)
    clusters = transcriber.cluster_embeddings(embeddings)
    protocol = f'protocols/' + audio.split('/')[-1].replace('.mp3', '.txt')
    transcriber.write_protocol(segments, clusters, protocol)

  if summarize:
    if not protocol:
        raise ValueError("No protocol file path provided for summarization.")
    print('Warning: Summary might not be very good.')

    # Authorize with Hugging Face API (You need to create the token in your HF account and request access to the meta llama3.2 model on HF: https://huggingface.co/meta-llama/Llama-3.2-3B)
    login(token=token)
    
    if audio:
      text = result['text'] # use text without diarization for the summary if it was just transcribed
    else:
      with open(protocol, 'r') as file:
        text = file.read().replace('\n', ' ')

    summarizer = Summarizer()
    summary = summarizer.summarize(text) 
    summarizer.save_summary(summary, protocol)


if __name__ == "__main__":
  # Parse command line arguments
  parser = argparse.ArgumentParser(description="Transcribe and optionally summarize audio files.")
  parser.add_argument("--audio", type=str, help="Path to the audio file (mp3)")
  parser.add_argument("--protocol", type=str, help="Path to the protocol file (txt)")
  parser.add_argument("--summarize", action="store_true", help="Summarize the transcribed text")
  parser.add_argument("--token", type=str, help="Hugging Face API token for summarization", required="--summarize" in sys.argv)
  args = parser.parse_args()
  main(args.audio, args.protocol, args.summarize, args.token)