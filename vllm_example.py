import sys
sys.path.append('third_party/Matcha-TTS')
from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm

def main():
    cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=True, load_vllm=True, fp16=True)
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
    for i in tqdm(range(100)):
        set_all_random_seed(i)
        for i, j in enumerate(cosyvoice.inference_zero_shot('I received a birthday gift from a friend from afar. The unexpected surprise and deep blessings filled my heart with sweet joy and my smile bloomed like a flower.', 'I hope you can do better than me in the future.', prompt_speech_16k, stream=False)):
            continue

if __name__=='__main__':
    main()
