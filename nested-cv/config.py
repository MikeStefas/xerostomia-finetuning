
# Paths
dataset_root = "../raw_aug"
model_name = "google/medgemma-1.5-4b-it"
cache_dir = "../patient_cache"

# Instruction prompt
instruction = """Analyze this intraoral image for visual signs of xerostomia. 
Reply ONLY with the word "Yes" or "No". No explanations."""

# Optuna configuration
N_OPTUNA_TRIALS = 10
RANDOM_SEED = 3407

