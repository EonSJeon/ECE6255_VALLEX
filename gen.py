from utils.prompt_making import make_prompt
from train_utils.icefall.utils import load_checkpoint

### Use given transcript
make_prompt(name="sample-jp", audio_prompt_path="sample-jp.wav",
                transcript="今日の予定を確認した。")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import utils.generation as gen

# download and load all models
gen.preload_models()

filename = "./checkpoints/final-checkpoint.pt"

load_checkpoint(filename, gen.model, None, None, None)

text_prompt = """
人生の中で何回かは大きな買い物をすることもあった。
"""
audio_array = gen.generate_audio(text_prompt, prompt="sample-jp", language='ja', accent='日本語')

write_wav("sample_original_cloned-jp.wav", SAMPLE_RATE, audio_array)
