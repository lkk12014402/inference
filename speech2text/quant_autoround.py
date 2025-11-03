import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, pipeline

from auto_round import AutoRoundMLLM, AutoRound

model_path = "./whisper/model"
bits, group_size, sym, act_bits = 8, -1, True, 8

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

## quantize the model
autoround = AutoRoundMLLM(model, tokenizer, processor,
                        bits=bits, group_size=group_size, sym=sym, act_bits=act_bits,
                        iters=0, 
                        layer_config={
                            "proj_out": {
                                "bits": bits, 
                                "group_size": group_size, 
                                "sym": sym, 
                                "act_bits": act_bits,
                            }
                        },
                    )
autoround.quantize()

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = "./whisper-large-v3-rtn"
autoround.save_quantized(output_dir, format='llm_compressor', inplace=True)
