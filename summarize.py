"""
Source: https://www.youtube.com/watch?v=fc7cAP5zrOY
"""

# Import packages
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain
from transformers import AutoTokenizer
import transformers
import torch

# For Testing
import warnings
warnings.filterwarnings("ignore")

class Summarizer:
  """
  A class used to generate summaries of meeting protocols using a pre-trained language model.

  Attributes
  ----------
  model_name : str
    The name of the pre-trained model to be used for text generation.
  tokenizer : AutoTokenizer
    The tokenizer associated with the pre-trained model.
  pipeline : transformers.pipeline
    The text generation pipeline configured with the specified model and tokenizer.
  llm : HuggingFacePipeline
    The language model pipeline used for generating summaries.
  template : str
    The template used for generating the summary prompt.
  prompt : PromptTemplate
    The prompt template object used to format the input text for the language model.
  llm_chain : LLMChain
    The chain object that links the prompt template and the language model.

  Methods
  -------
  summarize(protocol_path)
    Reads the protocol from the specified file path, generates a summary, and returns it.
  save_summary(summary, output_path)
    Saves the generated summary to the specified output file path.
  """

  def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    self.pipeline = transformers.pipeline(
      "text-generation",
      model=model_name,
      tokenizer=self.tokenizer,
      torch_dtype=torch.bfloat16,
      trust_remote_code=True,
      device_map="auto",
      max_new_tokens=500,
      truncation=True,
      do_sample=True,
      top_k=10,
      top_p=0.9, # default: None
      num_return_sequences=1,
      eos_token_id=self.tokenizer.eos_token_id
    )

    self.llm = HuggingFacePipeline(pipeline=self.pipeline, model_kwargs={'temperature': 0.9}) # default: 0.0
    self.delimiter = "```"
    self.template = \
    """
    Write a summary of the important content of the meeting protocol below.
    Include at least 10 bullet points covering all significant points.
    {self.delimiter + text + self.delimiter}
    BULLET POINTS SUMMARY:
    """
    
    self.prompt = PromptTemplate(template=self.template, input_variables=["text"])
    self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
    print('Initialized Summarizer.')


  def summarize(self, transcript):
    protocol = transcript.replace("\n", '')
    print('Creating summary... (this may take a while)')
    summary = self.llm_chain.run(protocol)
    print('Generated summary.')
    return summary


  def save_summary(self, summary, path):
    # Find the index of the first triple backtick and the last triple backtick
    start_index = summary.find(self.delimiter)
    end_index = summary.find(self.delimiter)

    # Extract the portion of the summary after the closing backticks
    cleaned_summary = summary[:start_index] + summary[end_index+len(self.delimiter):].strip()
    with open(path, 'r') as file:
      original = file.read()
    with open(path, 'w') as file:
      file.write('--- SUMMARY ---\nPrompt:\n' + cleaned_summary + '\n---------------\n\n' + original)
    print('Saved summary.')