import os
import torch
from PIL import Image
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bf16_supported
from datasets import load_from_disk, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import roc_auc_score
from utils import clear_gpu_memory
import config

def train_and_evaluate(train_patients, validation_patients, hyperparameters, epochs=3):
    clear_gpu_memory()

    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.model_name,
        load_in_4bit=False,
            use_gradient_checkpointing="unsloth",
            attn_implementation="sdpa",
        )

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            r=hyperparameters["r"],
            lora_alpha=hyperparameters["alpha"],
            target_modules=hyperparameters["target_modules"].split(","),
            lora_dropout=0,
        )

        # load and combine the cached datasets
        train_dataset = concatenate_datasets(
            [load_from_disk(os.path.join(config.cache_dir, patient)) for patient in train_patients]
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),
            train_dataset=train_dataset,
            args=SFTConfig(
                learning_rate=hyperparameters["lr"],
                weight_decay=hyperparameters["weight_decay"],
                num_train_epochs=epochs,
                max_steps=-1,
                warmup_ratio=0.1,
                per_device_train_batch_size=hyperparameters["per_device_batch"],
                gradient_accumulation_steps=1,
                bf16=is_bf16_supported(),
                optim="adamw_8bit",
                output_dir="outputs_optuna",
                report_to="none",
                dataset_num_proc=1,
                dataloader_num_workers=0,
                max_seq_length=1024,
            ),
        )

        trainer.train()

        # switch to inference mode for evaluation
        FastVisionModel.for_inference(model)
        y_true, y_probabilities, results = [], [], []

        yes_token = tokenizer.tokenizer.encode("Yes", add_special_tokens=False)[-1]
        no_token = tokenizer.tokenizer.encode("No", add_special_tokens=False)[-1]

        for patient in validation_patients:
            patient_path = os.path.join(config.dataset_root, patient)
            label = 1 if patient.startswith("X") else 0

            # only use original (non-augmented) images for evaluation
            images = sorted([filename for filename in os.listdir(patient_path) if filename.lower().endswith((".jpg", ".jpeg", ".png")) and "aug" not in filename])
            images = images[:4]

            for image_name in images:
                image = Image.open(os.path.join(patient_path, image_name)).convert("RGB")
                
                with torch.no_grad():
                    input_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": config.instruction}]}],
                        add_generation_prompt=True,
                    )
                    inputs = tokenizer(image, input_text, return_tensors="pt", add_special_tokens=False).to("cuda")

                    outputs = model(**inputs, use_cache=False)
                    logits = outputs.logits[:, -1, :]
                    
                    probability_yes = torch.softmax(logits[0, [yes_token, no_token]], dim=-1)[0].item()
                    y_probabilities.append(probability_yes)
                    y_true.append(label)
                    results.append({
                        "file_name": f"{patient}/{image_name}",
                        "y_prob": probability_yes,
                        "y_true": label
                    })
                    
                    # explicit cleanup of large tensors in loop
                    del inputs, outputs, logits

        auc = roc_auc_score(y_true, y_probabilities)
        return auc, results

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA OOM caught! Error: {e}")
            return 0.0, [] # Return failure score
        else:
            raise e
    finally:
        # aggressive cleanup
        if 'trainer' in locals(): del trainer
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        if 'train_dataset' in locals(): del train_dataset
        clear_gpu_memory()
