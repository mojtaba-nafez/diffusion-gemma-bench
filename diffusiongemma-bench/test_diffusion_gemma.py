import torch

from transformers import (
    AutoConfig,
    DiffusionGemmaForBlockDiffusion,
)

model_name = "google/diffusiongemma-26B-A4B-it"

config = AutoConfig.from_pretrained(model_name)

config.text_config.rm_self_attention = True

model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    model_name,
    config=config,
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

model.eval()

# Prompt
message = [
    {"role": "user", "content": "Why is the sky blue?"}
]

# Process input
input_ids = processor.apply_chat_template(
    message,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
output = model.generate(**input_ids, max_new_tokens=512)

# Parse output
text = processor.decode(output[0], skip_special_tokens=False)
print(text)

# messages = [
#     {
#         "role": "user",
#         "content": "What is 2 + 2? Explain briefly.",
#     }
# ]


# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt",
# )

# inputs = {
#     k: v.to(model.device) if hasattr(v, "to") else v
#     for k, v in inputs.items()
# }


# with torch.inference_mode():
#     output = model.generate(
#         **inputs,
#         max_new_tokens=128,
#     )


# sequences = (
#     output.sequences
#     if hasattr(output, "sequences")
#     else output
# )


# print(
#     processor.decode(
#         sequences[0],
#         skip_special_tokens=False,
#     )
# )