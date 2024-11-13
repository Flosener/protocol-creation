"""
Credit: Dwarkesh's Patel (https://x.com/dwarkesh_sp/status/1579672641887408129)
Source: https://www.youtube.com/watch?v=MVW746z8y_I
"""

# Import packages
import whisper
import datetime
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pydub import AudioSegment

# For Testing
import warnings
warnings.filterwarnings("ignore")

class Transcriber:
  """
  A class used to transcribe audio files, generate speaker embeddings, cluster them, and write a protocol.

  Attributes
  ----------
  num_speakers : int
    The number of speakers in the audio file.
  language : str
    The language of the audio file.
  model_size : str
    The size of the model to be used for transcription.
  summarize : bool
    Whether to summarize the transcription.
  model_name : str
    The name of the model to be used for transcription.
  model : whisper.Model
    The Whisper model used for transcription.
  embedding_model : PretrainedSpeakerEmbedding
    The model used for generating speaker embeddings.
  audio : Audio
    The Audio object used for processing audio files.

  Methods
  -------
  transcribe(file_path)
    Transcribes the audio file at the given file path.
  generate_embeddings(segments, duration, file_path)
    Generates embeddings for the given segments of the audio file.
  segment_embedding(segment, duration, file_path)
    Generates an embedding for a single segment of the audio file.
  cluster_embeddings(embeddings)
    Clusters the given embeddings into the number of speakers.
  write_protocol(segments, labels, file_path_out, max_line_length=80)
    Writes the transcription protocol to a file.
  time(secs)
    Converts seconds to a timedelta object.
  wrap_text(text, max_length)
    Wraps the given text to the specified maximum line length.
  """

  def __init__(self, num_speakers, language, model_size):
    self.num_speakers = num_speakers
    self.language = language
    self.model_size = model_size
    self.model_name = model_size
    if language == 'English' and model_size != 'large':
      self.model_name += '.en'
    self.model = whisper.load_model(self.model_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    self.embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device='cuda' if torch.cuda.is_available() else 'cpu')
    self.audio = Audio()
    print('Initialized Transcriber.')

  def transcribe(self, file_path):
    print('Transcribing audio... (this may take a while)')
    result = self.model.transcribe(file_path)
    print('Transcribed audio.')
    audio = AudioSegment.from_mp3(file_path)
    duration = len(audio) / 1000.0  # duration in seconds
    return result, duration

  def generate_embeddings(self, segments, duration, file_path):
    embeddings = np.zeros(shape=(len(segments), 192))
    for i, segment in enumerate(segments):
      embeddings[i] = self.segment_embedding(segment, duration, file_path)
    embeddings = np.nan_to_num(embeddings)
    print('Generated embeddings.')
    return embeddings

  def segment_embedding(self, segment, duration, file_path):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = self.audio.crop(file_path, clip)
    #print(f'Generated embedding for segment: {start} to {end}.')
    return self.embedding_model(waveform[None])

  def cluster_embeddings(self, embeddings):
    clustering = AgglomerativeClustering(self.num_speakers).fit(embeddings)
    print('Clustered embeddings.')
    return clustering.labels_

  def write_protocol(self, segments, labels, file_path_out, max_line_length=None):
    for i in range(len(segments)):
      segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    with open(file_path_out, "w") as f:
      for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
          f.write(('\n\n' if i != 0 else '') + segment["speaker"] + ' ' + str(self.time(segment["start"])) + '\n')
        text = segment["text"][1:]
        if max_line_length:
          text = self.wrap_text(text, max_line_length)
        f.write(text + ' ')
    print('Written protocol file.')

  @staticmethod
  def time(secs):
    return datetime.timedelta(seconds=round(secs))

  @staticmethod
  def wrap_text(text, max_length):
    words = text.split()
    lines, current_line = [], ""
    for word in words:
      if len(current_line) + len(word) + 1 > max_length:
        lines.append(current_line)
        current_line = word
      else:
        current_line += (" " if current_line else "") + word
    if current_line:
      lines.append(current_line)
    return "\n".join(lines)