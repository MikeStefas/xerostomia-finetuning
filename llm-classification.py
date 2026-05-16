from unsloth import FastVisionModel
from PIL import Image
import torch
import os
import pandas as pd
import torch.nn.functional as F
import time

instruction_zero_shot = (
    """Analyze this intraoral image for visual signs of xerostomia. 
    Reply ONLY with the word "Yes" or "No". No explenation"""
)

def diagnose_image_soft(image_path, instruction, model, tokenizer):
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :] 

    yes_token_id = tokenizer.tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_token_id = tokenizer.tokenizer.encode("No", add_special_tokens=False)[-1]
    yes_logit = logits[:, yes_token_id]
    no_logit = logits[:, no_token_id]

    probs = F.softmax(torch.cat([no_logit, yes_logit]), dim=-1)
    
    prob_yes = probs[1].item() 
    
    label = "Yes" if prob_yes >= 0.5 else "No"
    
    return prob_yes, label

if __name__ == "__main__":
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "google/medgemma-1.5-4b-it", 
        load_in_4bit = False,
        use_gradient_checkpointing = "unsloth",
    )
    
    FastVisionModel.for_inference(model)

    start = time.time()

    res_pd = pd.DataFrame(columns=["image_path", "prob_yes", "result"])
    parent_folder = "raw"
    
    output_csv = "class_gemma4_results_16b.csv"

    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if not os.path.isdir(subfolder_path): continue
        
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            
            prob_yes, label = diagnose_image_soft(image_path, instruction_zero_shot, model, tokenizer)
            
            print(f"Path: {image_name} | Prob: {prob_yes:.4f} | Label: {label}")
            
            res_pd.loc[len(res_pd)] = [image_path, prob_yes, label]
            res_pd.to_csv(output_csv, index=False)
        break
    end = time.time()
    print(end - start)
