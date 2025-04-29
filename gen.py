from utils.prompt_making import make_prompt
from train_utils.icefall.utils import load_checkpoint

### Use given transcript
make_prompt(name="indian", audio_prompt_path="indian_accent.wav",
                transcript="not wanted my weaknesses to be detected by Bhabaprasad")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import utils.generation as gen

# download and load all models
gen.preload_models()

filename = "./checkpoints/vallex-checkpoint.pt"

load_checkpoint(filename, gen.model, None, None, None)

text_prompt = """
not wanted my weaknesses to be detected by Bhabaprasad
"""
audio_array = gen.generate_audio(text_prompt, prompt="indian", language='en', accent='English')

write_wav("sample_indian.wav", SAMPLE_RATE, audio_array)
