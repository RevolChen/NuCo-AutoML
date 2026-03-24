import pandas as pd
import os

datasets = {
    'CIFD': {'task': 'classification'},
    'FJPP': {'task': 'classification'},
    'IMDB': {'task': 'regression'},
    'PPC': {'task': 'regression'},
    'PAP': {'task': 'multiclassification'}
}

output_dir = "new_features"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

base_path = "tests/data_ji"
seed_path = "tests/data_ji/seed52"

for ds_name, info in datasets.items():
    task = info['task']
    # Original data paths
    orig_train_path = os.path.join(base_path, task, ds_name, "train_split.csv")
    orig_test_path = os.path.join(base_path, task, ds_name, "test_split.csv")
    
    # Generated data paths
    gen_train_path = os.path.join(seed_path, task, "outputs_CAAFE", f"{ds_name}_original_CAAFE_train.csv")
    gen_test_path = os.path.join(seed_path, task, "outputs_CAAFE", f"{ds_name}_original_CAAFE_test.csv")
    
    if not os.path.exists(gen_train_path):
        print(f"Warning: Generated file not found for {ds_name}")
        continue
        
    # Read data
    try:
        df_orig_train = pd.read_csv(orig_train_path)
        df_gen_train = pd.read_csv(gen_train_path)
        
        df_orig_test = pd.read_csv(orig_test_path)
        df_gen_test = pd.read_csv(gen_test_path)
    except Exception as e:
        print(f"Error reading files for {ds_name}: {e}")
        continue
    
    # Identify new columns
    # We only care about columns present in generated data but NOT in original data
    new_cols = [c for c in df_gen_train.columns if c not in df_orig_train.columns]
    
    print(f"Dataset {ds_name}: New features found: {new_cols}")
    
    if new_cols:
        # Save new features
        df_new_train = df_gen_train[new_cols]
        df_new_test = df_gen_test[new_cols]
        
        train_save_path = os.path.join(output_dir, f"{ds_name}_new_features_train.csv")
        test_save_path = os.path.join(output_dir, f"{ds_name}_new_features_test.csv")
        
        df_new_train.to_csv(train_save_path, index=False)
        df_new_test.to_csv(test_save_path, index=False)
        print(f"Saved to {train_save_path} and {test_save_path}")
    else:
        print(f"Dataset {ds_name}: No new features found.")

print("All done.")
