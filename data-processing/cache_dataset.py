import os
import torch
from PIL import Image
from datasets import Dataset
import random

dataset_root = "../raw_aug"
cache_dir = "../patient_cache"
instruction = """Analyze this intraoral image for visual signs of xerostomia. 
Reply ONLY with the word "Yes" or "No". No explanations."""

# Create the cache directory if it doesn't exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

# Get all patient folders
all_patients = []
for f in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root, f)):
        all_patients.append(f)
all_patients.sort()

print(f"Found {len(all_patients)} patients. Starting caching...")

for patient in all_patients:
    patient_path = os.path.join(dataset_root, patient)
    patient_save_path = os.path.join(cache_dir, patient)
    
    
    
    # Label is Yes if folder starts with X, otherwise No
    if patient.startswith("X"):
        label = "Yes"
    else:
        label = "No"
        
    samples = []
    for img_name in os.listdir(patient_path):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
            
        img_path = os.path.join(patient_path, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            samples.append({
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
                    {"role": "assistant", "content": [{"type": "text", "text": label}]}
                ],
                "image": image,
                "patient_name": patient
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Create and save the dataset for this participant
    if samples:
        ds = Dataset.from_list(samples)
        ds.save_to_disk(patient_save_path)
        print(f"   Successfully cached {len(samples)} images for {patient}")
    else:
        print(f"   Warning: No images found for {patient}")

print("\nCaching complete! You can now load these individually in your CV loop.")
