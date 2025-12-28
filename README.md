# Finetuning LLaMA 3.1 (8B) with Unsloth — Alpaca-style SFT

This repository demonstrates how to perform efficient supervised finetuning (SFT) of Meta LLaMA-3.1 8B using the Unsloth framework (FastLanguageModel) and LoRA-style parameter-efficient finetuning. The included Jupyter notebook (`llama-3.1-8b-finetuned_alpaca.ipynb`) walks through loading a quantized model, applying PEFT/LoRA, preparing the Alpaca dataset, training with TRL's SFTTrainer, running inference, and exporting/quantizing to GGUF for llama.cpp.

Notebook (reference): [llama-3.1-8b-finetuned_alpaca.ipynb](https://github.com/abhilashpanda04/Finetuning-Llama-3.1-Unsloth/blob/58af4e8ddf5d685f0ebeedf431f5b9365080ed43/llama-3.1-8b-finetuned_alpaca.ipynb)

---

Table of contents
- Overview
- Requirements
- Quick start
- Notebook walkthrough
  - 1) Load model (4-bit / Unsloth)
  - 2) Apply PEFT / LoRA
  - 3) Prepare Alpaca dataset (yahma/alpaca-cleaned)
  - 4) Configure SFTTrainer and train
  - 5) Inference (Unsloth accelerated)
  - 6) Save / Push and GGUF conversion
- Example commands and snippets
- Tips & troubleshooting
- License & credits

---

Overview
--------
This notebook uses Unsloth's FastLanguageModel to:
- load a LLaMA-3.1 8B model (optionally in 4-bit),
- patch the model for memory / speed improvements,
- convert it into a PEFT (LoRA) model,
- finetune on the Alpaca-cleaned dataset with TRL's SFTTrainer,
- run accelerated inference and export formats (GGUF / llama.cpp).

Unsloth provides optional optimizations (gradient checkpointing, unsloth patching) to reduce VRAM and speed training/inference.

Requirements
------------
Minimum recommended environment:
- CUDA-enabled GPU with ample memory (example used: NVIDIA RTX A6000 ~48 GB)
- Linux (tested)
- Python 3.10+
- PyTorch 2.x built with CUDA (matching your CUDA toolkit)
- Transformers (>= 4.44.0 as in notebook; check Unsloth compatibility)
- unsloth (Unsloth project; provides FastLanguageModel)
- trl (TRL library for SFTTrainer)
- peft
- datasets (Hugging Face datasets)
- ipywidgets / jupyterlab (optional but recommended for notebook UX)

Example conda environment
```bash
conda create -n unsloth_env python=3.10 -y
conda activate unsloth_env

# Install PyTorch + CUDA (choose matching CUDA version)
# Example for CUDA 12.1 (adjust according to your machine):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install HF ecosystem + TRL + PEFT + Unsloth (adjust versions as needed)
pip install transformers datasets accelerate trl peft safetensors
# Unsloth may not be on PyPI; follow its repo installation instructions:
# pip install git+https://github.com/unslothai/unsloth.git
```

Quick start
-----------
1. Clone the repository (or open the notebook in Colab/your local Jupyter).
2. Install environment dependencies (see above).
3. Open `llama-3.1-8b-finetuned_alpaca.ipynb` and run cells in order.
4. Edit hyperparameters (LoRA rank r, batch sizes, max_steps, etc.) as needed.

Notebook walkthrough
--------------------

1) Load model (4-bit / Unsloth)
- Use Unsloth's FastLanguageModel.from_pretrained to load a LLaMA 3.1 8B model.
- You can optionally load a 4-bit variant (Unsloth supports specifying `load_in_4bit=True`).
- Example parameters in the notebook:
  - max_seq_length = 2048
  - load_in_4bit = True
  - four_bit_models = ["unsloth/Meta-Llama-3.1-8B-bnb-4bit"]

2) Apply PEFT / LoRA
- `FastLanguageModel.get_peft_model` is used to convert the model to a LoRA/PEFT instance.
- Notebook example hyperparameters:
  - r = 16 (LoRA rank; recommended values: 8,16,32,64)
  - target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
  - lora_alpha = 16, lora_dropout = 0, bias = "none"
  - use_gradient_checkpointing = "unsloth" (Unsloth-optimized checkpointing)
- This makes training memory-efficient and trains only the LoRA parameters.

3) Prepare Alpaca dataset
- The notebook uses `datasets.load_dataset("yahma/alpaca-cleaned", split="train")`.
- It maps a formatting function which constructs instruction+input+response text per Alpaca style and appends EOS token.
- Ensure dataset download and mapping succeed (sufficient disk and network).

4) Configure SFTTrainer and train
- Uses `trl.SFTTrainer` with a `transformers.TrainingArguments` object.
- Example training settings:
  - per_device_train_batch_size = 2
  - gradient_accumulation_steps = 4 (effective batch size = 8)
  - max_steps = 60 (overrides num_train_epochs)
  - learning_rate = 2e-4
  - optim = "adamw_8bit"
  - bf16 / fp16 chosen based on hardware (noted using `unsloth.is_bfloat16_supported()` in notebook).
- Trainer prints step-by-step training losses and progress.

5) Inference (Unsloth accelerated)
- Use `FastLanguageModel.for_inference(model)` to enable Unsloth's native inference optimizations.
- Build prompt using the same Alpaca instruction template.
- Two generation examples are shown:
  - Batched generation storing outputs via `model.generate(...)`
  - Streamed generation with `TextStreamer` for interactive token streaming

6) Save / Push and GGUF conversion
- Save LoRA weights locally:
  - `model.save_pretrained("lora_model")`
  - `tokenizer.save_pretrained("lora_model")`
- Push to Hugging Face Hub:
  - `model.push_to_hub("username/llama3.1-8b-alpaca_lora_model", token="HF_TOKEN")`
  - `tokenizer.push_to_hub("username/token_llama3.1-8b-alpaca_lora_model", token="HF_TOKEN")`
- GGUF / llama.cpp conversion:
  - Unsloth supports exporting to GGUF and common quantization methods (q8_0, q4_k_m, q5_k_m, etc.)
  - Example functions in notebook: `model.save_pretrained_gguf(...)`, `model.push_to_hub_gguf(...)`

Example commands & snippets
--------------------------
- Load the four-bit model
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

- Create LoRA/PEFT model
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

- Train (example)
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=60,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs",
    ),
)
trainer.train()
```

- Simple inference
```python
FastLanguageModel.for_inference(model)
prompt = alpaca_prompt.format("Continue the fibonacci sequence.", "1, 1, 2, 3, 5, 8", "")
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.batch_decode(outputs)[0])
```

Tips & troubleshooting
----------------------
- Out of memory: reduce per-device batch size, lower LoRA rank `r`, use Unsloth optimizations (`use_gradient_checkpointing="unsloth"`), or load smaller quantized model.
- Installation: Unsloth may require specific Transformers/PyTorch versions. Check Unsloth's repo for compatibility.
- IPython warnings about tqdm/iProgress: install/upgrade `ipywidgets` and `jupyter`.
- If `is_bfloat16_supported()` is True, prefer bf16 (better numerical range on supported hardware).
- When saving/pushing to HF Hub, set `token = os.environ["HF_TOKEN"]` and keep tokens secret.
- Always append EOS token when preparing sequence-to-sequence style training prompts to prevent runaway generations.

Notes and caveats
-----------------
- This notebook is a demonstration — tune hyperparameters (steps, lr, batch size) for real training runs.
- The notebook uses small `max_steps` and example settings suited for a quick demo; for production, adapt dataset size, epochs, and evaluation.
- Respect licensing and model usage policies for Meta LLaMA and any checkpoint you use.

License & credits
-----------------
- Code and approach adapted from Unsloth (https://github.com/unslothai/unsloth) and TRL.
- Dataset: `yahma/alpaca-cleaned` (Hugging Face).
- This README and example code are provided for educational/demo purposes. Check and comply with licenses for model checkpoints and third-party libraries.

Acknowledgements
----------------
- Unsloth team for providing FastLanguageModel and speed/V RAM optimizations.
- Hugging Face for datasets, transformers and TRL ecosystem.
- Alpaca community and dataset maintainers.
