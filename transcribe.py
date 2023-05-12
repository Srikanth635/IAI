import whisper
import torch
import time
# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = """they are right, right, right, from prepping with our 10 points. Right, necessary. Okay. Right, just hand.
Yeah. Yeah, what I need from you is if you can, even before the hands, can you send, I shift the constraints based map, or can you create a pull request?"""

print(sent_tokenize(text))


# # torch.cuda.is_available()
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)
#
# wav = '16_17_55.wav'
# # Whisper Params
# model = whisper.load_model('base',device=DEVICE)
# result = model.transcribe(wav)
# print(result.keys())
# print(result["text"])
# print(result["segments"][0])
# print(result["segments"])
# print(result["language"])