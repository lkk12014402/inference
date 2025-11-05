import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "openai/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Configure processor the dataset task.
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[32:{NUM_CALIBRATION_SAMPLES}]",
)
print(ds)
print(ds[0])


def preprocess(example):
    print(f"=========================================================")
    print(example)
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
        "text": " " + example["text"].capitalize(),
    }


ds = ds.map(preprocess, remove_columns=ds.column_names)
print(ds)

print(model.dtype)

# Process inputs.
def process(sample):
    inputs = processor(
        audio=sample["array"],
        sampling_rate=sample["sampling_rate"],
        text=sample["text"],
        add_special_tokens=True,
        return_tensors="pt",
    )

    inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    inputs["decoder_input_ids"] = inputs["labels"]
    del inputs["labels"]

    return inputs


ds = ds.map(process, remove_columns=ds.column_names)


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    # print(batch)
    #exit()
    new_batch = {}
    for key, value in batch[0].items():
        if key == "input_features":
            new_batch[key] = torch.tensor(value, dtype=model.dtype)
        if key == "decoder_input_ids":
            new_batch[key] = torch.tensor(value)
        else:
            new_batch[key] = torch.tensor(value, dtype=model.dtype)
    return new_batch
    # return {key: torch.tensor(value, dtype=model.dtype) for key, value in batch[0].items()}


recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)



"""
# Recipe
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)
"""

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample_features = next(iter(ds))["input_features"]
sample_decoder_ids = [processor.tokenizer.prefix_tokens]
sample_input = {
    "input_features": torch.tensor(sample_features, dtype=model.dtype).to(model.device),
    "decoder_input_ids": torch.tensor(sample_decoder_ids).to(model.device),
}
output = model.generate(**sample_input, language="en")
print(processor.batch_decode(output, skip_special_tokens=True))
print("==========================================\n\n")
# that's where you have a lot of windows in the south no actually that's passive solar
# and passive solar is something that was developed and designed in the 1960s and 70s
# and it was a great thing for what it was at the time but it's not a passive house

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-nvfp4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
