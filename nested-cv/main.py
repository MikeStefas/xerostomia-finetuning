import os
import optuna
import json
import csv
import numpy as np
import config
from utils import clear_gpu_memory, get_folds, get_all_patients, log
from modeling import train_and_evaluate

def main():
    # Get all patients and split them into 5 groups for the outer loop
    all_patients = get_all_patients()
    outer_folds = get_folds(all_patients, 5)

    all_outer_results = []
    all_best_params = []

    # OUTER LOOP
    for fold_idx in range(len(outer_folds)):
        train_pts, test_pts = outer_folds[fold_idx]
        
        log(f"\n{'='*60}\nSTARTING OUTER FOLD {fold_idx + 1}/5\n{'='*60}")
        
        # Split the training data again into 4 parts for tuning hyperparameters
        inner_folds = get_folds(train_pts, 4)

        def objective(trial):
            clear_gpu_memory()

            # Define the search space for different hyperparameters
            params = {
                "lr":               trial.suggest_float("lr", 5e-6, 5e-4, log=True),
                "r":                trial.suggest_categorical("r", [8, 16, 32, 64]),
                "alpha":            trial.suggest_categorical("alpha", [16, 32, 64]),
                "weight_decay":     trial.suggest_float("weight_decay", 0.01, 0.1),
                "per_device_batch": trial.suggest_categorical("per_device_batch", [4, 8, 16]),
                "target_modules":   trial.suggest_categorical("target_modules", [
                    "q_proj,v_proj",
                    "q_proj,k_proj,v_proj,o_proj",
                    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                ]),
            }

            scores = []
            
            # Run 4-fold CV for these specific parameters
            for i in range(len(inner_folds)):
                i_train, i_val = inner_folds[i]
                log(f"  [Outer {fold_idx+1}][Trial {trial.number}] Checking Inner Fold {i+1}/4...")
                
                # Test the params for 1 epoch for speed
                auc, _ = train_and_evaluate(i_train, i_val, params, epochs=1)
                scores.append(auc)
                
                # Report to Optuna so it can stop early if the trial is failing
                trial.report(auc, i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Return the average score of the 4 inner folds
            return np.mean(scores)

        # Start the Optuna search for this outer fold
        log(f"--- Running Optuna optimization for outer fold {fold_idx + 1} ---")
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        )
        study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS)

        # Get the best parameters we found
        best_p = study.best_params
        all_best_params.append(best_p)
        log(f"Best settings found: {best_p}")

        # Save best parameters to a file
        with open(f"best_params_fold_{fold_idx + 1}.json", "w") as f:
            json.dump(best_p, f, indent=4)

        # FINAL STEP: Train on the full training set 1 last time and test on actual held-out patients
        log(f"--- Final evaluation for Outer Fold {fold_idx + 1} ---")
        test_auc, test_results = train_and_evaluate(train_pts, test_pts, best_p, epochs=3)
        all_outer_results.append(test_auc)
        log(f"Final Test AUC for Fold {fold_idx + 1}: {test_auc:.4f}")

        # Save the detailed predictions to a CSV
        csv_name = f"test_results_fold_{fold_idx + 1}.csv"
        with open(csv_name, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "y_pred", "y_true"])
            writer.writeheader()
            for r in test_results:
                writer.writerow({
                    "file_name": r["file_name"],
                    "y_pred": r["y_prob"],
                    "y_true": r["y_true"]
                })
        log(f"Saved predictions to {csv_name}")

    # Save a summary of all best params across folds
    with open("all_best_params.json", "w") as f:
        json.dump(all_best_params, f, indent=4)
    log("Saved all best params to all_best_params.json")

    # Print the grand total average
    log(f"\n{'='*60}\nALL FOLDS COMPLETE\n{'='*60}")
    log(f"Mean performance: {np.mean(all_outer_results):.4f} +/- {np.std(all_outer_results):.4f}")

if __name__ == "__main__":
    main()
