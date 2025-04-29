# from utils.prompt_making import make_prompt
from train_utils.icefall.utils import load_checkpoint
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn

import logging
# ### Use given transcript
# make_prompt(name="sample-jp", audio_prompt_path="sample-jp.wav",
#                 transcript="今日の予定を確認した。")

from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import utils.generation as gen

# download and load all models
gen.preload_models()

filename = "./checkpoints/final-checkpoint.pt"

load_checkpoint(filename, gen.model, None, None, None)


# (1) Freeze your base model if you haven’t already
for param in gen.model.parameters():
    param.requires_grad = False

# (2) Explicitly list all the Linear layers you want LoRA’d
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # match every layer you printed above:
    target_modules=[
        # AR decoder
        "ar_decoder.layers.*.self_attn.out_proj",
        "ar_decoder.layers.*.linear1",
        "ar_decoder.layers.*.linear2",
        "ar_predict_layer",
        # NAR decoder
        "nar_decoder.layers.*.self_attn.out_proj",
        "nar_decoder.layers.*.linear1",
        "nar_decoder.layers.*.linear2",
        "nar_decoder.layers.*.norm1.project_layer",
        "nar_decoder.layers.*.norm2.project_layer",
        "nar_decoder.norm.project_layer",
        # the NAR output heads
        "nar_predict_layers.*",
    ],
)

from types import MethodType

# stub for HF generation API hooks
def _stub_prepare_inputs_for_generation(self, input_ids, **kwargs):
    return {"x": input_ids, **kwargs}

def _stub_prepare_encoder_decoder_kwargs_for_generation(self, **kwargs):
    # return exactly what PEFT will want to pass into forward
    return kwargs

# attach them to your base model
gen.model.prepare_inputs_for_generation = MethodType(
    _stub_prepare_inputs_for_generation, gen.model
)
gen.model._prepare_encoder_decoder_kwargs_for_generation = MethodType(
    _stub_prepare_encoder_decoder_kwargs_for_generation, gen.model
)

# now this will succeed
model = get_peft_model(gen.model, lora_config)
model.print_trainable_parameters()
