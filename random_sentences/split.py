import os
import random

def split_data(file_path, train_ratio=0.7, test_ratio=0.15, dev_ratio=0.15, seed=42):
    random.seed(seed)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * train_ratio)
    test_end = train_end + int(total * test_ratio)
    
    train_data = lines[:train_end]
    test_data = lines[train_end:test_end]
    dev_data = lines[test_end:]
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = "splits"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"{base_name}_train.txt"), 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    with open(os.path.join(output_dir, f"{base_name}_test.txt"), 'w', encoding='utf-8') as f:
        f.writelines(test_data)
    
    with open(os.path.join(output_dir, f"{base_name}_dev.txt"), 'w', encoding='utf-8') as f:
        f.writelines(dev_data)
    
    print(f"Data split completed for {file_path}.")
    print(f"Train: {len(train_data)} lines, Test: {len(test_data)} lines, Dev: {len(dev_data)} lines")
    print(f"Splits saved in the 'splits' directory.")

files = ["msmarco.txt"]

for file in files:
    split_data(file, train_ratio=0.7, test_ratio=0.15, dev_ratio=0.15, seed=42)