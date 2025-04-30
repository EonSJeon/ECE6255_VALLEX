# from utils.prompt_making import make_prompt
# from train_utils.icefall.utils import load_checkpoint

# ### Use given transcript
# make_prompt(name="indian", audio_prompt_path="sample_indian.wav",
#                 transcript="not wanted my weaknesses to be detected by Bhabaprasad")

# from utils.generation import SAMPLE_RATE, generate_audio, preload_models
# from scipy.io.wavfile import write as write_wav
# import utils.generation as gen

import pandas as pd

df = pd.read_csv("./test_dataset/en/english_transcripts.csv")

for index, row in df.iterrows():
    print(row['id'])
    print(row['text'])
    # print(row['romaji'])
    # print(row['english'])
    print("--------------------------------")

# # download and load all models
# gen.preload_models()

# pure_checkpoint = "./checkpoints/checkpoint-30-pure.pt"
# cloned_checkpoint = "./checkpoints/checkpoint-30-cloned.pt"

# load_checkpoint(pure_checkpoint, gen.model, None, None, None)

# text_prompt = """
# not wanted my weaknesses to be detected by Bhabaprasad"
# """
# audio_array = gen.generate_audio(text_prompt, prompt="indian", language='en', accent='English')

# write_wav("sample_indian_cloned-pure.wav", SAMPLE_RATE, audio_array)
