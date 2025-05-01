# from utils.prompt_making import make_prompt
from train_utils.icefall.utils import load_checkpoint

# ### Use given transcript
# make_prompt(name="krishna_en", audio_prompt_path="sample_krishna_en.wav",
#                 transcript="This made a strange noise.")
# make_prompt(name="krishna_jp", audio_prompt_path="sample_krishna_jp.wav",
#                 transcript="お名前は何ですか？")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import utils.generation as gen



# download and load all models
gen.preload_models()


checkpoint_paths = {
    "original":   "./checkpoints/vallex-checkpoint.pt",
    "pure":       "./checkpoints/checkpoint-30-pure.pt",
    "LoRA_whole": "./checkpoints/checkpoint-30-whole.pt",
    "LoRA_AR":    "./checkpoints/checkpoint-30-AR.pt",
    "LoRA_NAR":   "./checkpoints/checkpoint-30-NAR.pt",
}

opt = "LoRA_whole"
text_prompt = """
This made a strange noise."
"""

for opt in checkpoint_paths.keys():
    load_checkpoint(checkpoint_paths[opt], gen.model, None, None, None)

    audio_array = gen.generate_audio(text_prompt, prompt="cafe", language='en', accent='English')

    write_wav(f"sample_cafe_cloned-{opt}.wav", SAMPLE_RATE, audio_array)
