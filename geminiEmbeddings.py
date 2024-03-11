import textwrap
import numpy as np
import pandas as pd
import os 
import google.generativeai as genai
import google.ai.generativelanguage as glm

# Used to securely store your API key

from IPython.display import Markdown

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)
 