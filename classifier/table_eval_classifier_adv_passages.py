import json
import os
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm

classifier_directories = {
    'all_injections': '../attack_results/'
}

RELEVANT_FILE_SUFFIX = '_targeted_classifier.jsonl'

def extract_injection_type_and_dataset(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = base_name.replace(RELEVANT_FILE_SUFFIX, "")

    parts = name_without_ext.split('_')

    injection_type = ""
    dataset = ""

    if name_without_ext.startswith('keyword_injection'):
        injection_type = 'Keyword Injection'
        dataset = parts[4] if len(parts) > 4 else 'Unknown'
    elif name_without_ext.startswith('query_injection'):
        injection_type = 'Query Injection'
        dataset = parts[4] if len(parts) > 4 else 'Unknown'
    elif name_without_ext.startswith('random_sentence_injection'):
        injection_type = 'Sentence Injection (MSMARCO)'
        dataset = parts[6] if len(parts) > 6 else 'Unknown'
    elif name_without_ext.startswith('targeted_sentence_injection'):
        injection_type = 'Sentence Injection (Toxigen)'
        dataset = parts[6] if len(parts) > 6 else 'Unknown'
    else:
        injection_type = 'Unknown Injection'
        dataset = 'Unknown Dataset'

    dataset = dataset.replace('-', ' ').title()

    return injection_type, dataset

def main():
    aggregated_counts = defaultdict(lambda: defaultdict(lambda: {"total": 0, "classification_1": 0}))

    all_jsonl_files = []
    for injection_category, dir_path in classifier_directories.items():
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}. Skipping.")
            continue
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(RELEVANT_FILE_SUFFIX):
                    all_jsonl_files.append(os.path.join(root, file))

    if not all_jsonl_files:
        print("No relevant JSONL files found in the specified directories.")
        return

    for file_path in tqdm(all_jsonl_files, desc="Processing classifier files"):
        injection_type, dataset = extract_injection_type_and_dataset(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        print(f"Invalid JSON at line {line_num} in {file_path}. Skipping line.")
                        continue

                    attack_ids = entry.get('attack_ids', "")
                    if isinstance(attack_ids, list):
                        if "control" in attack_ids:
                            continue
                    elif isinstance(attack_ids, str):
                        if "control" in attack_ids:
                            continue

                    include_entry = False
                    if injection_type in (
                        'Query Injection',
                        'Keyword Injection',
                        'Sentence Injection (MSMARCO)',
                        'Sentence Injection (Toxigen)'
                    ):
                        include_entry = True

                    if not include_entry:
                        continue

                    classification = entry.get('classification', 0)

                    if classification not in [0, 1]:
                        print(f"Unexpected classification value '{classification}' at line {line_num} in {file_path}. Skipping entry.")
                        continue

                    aggregated_counts[injection_type][dataset]["total"] += 1
                    if classification == 1:
                        aggregated_counts[injection_type][dataset]["classification_1"] += 1

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    table_data = []

    for injection_type, datasets in aggregated_counts.items():
        for dataset, counts in datasets.items():
            total = counts["total"]
            classification_1 = counts["classification_1"]
            classification_0 = total - classification_1
            proportion_0 = (classification_0 / total * 100) if total > 0 else 0.0

            table_data.append({
                'Injection Type': injection_type,
                'Dataset': dataset,
                'Classification=0 (%)': f"{proportion_0:.2f}"
            })

    if not table_data:
        print("No data to display after processing all files.")
        return

    required_keys = ['Injection Type', 'Dataset', 'Classification=0 (%)']
    for idx, row in enumerate(table_data, 1):
        missing_keys = [key for key in required_keys if key not in row]
        if missing_keys:
            print(f"Row {idx} is missing keys: {missing_keys}")

    table_data = sorted(table_data, key=lambda x: (x['Injection Type'], x['Dataset']))

    print("Headers:", required_keys)
    print("Sample Data:")
    for row in table_data[:5]:  # Print first 5 rows
        print(row)

    table = tabulate(table_data, headers="keys", tablefmt="grid")

    print("\nProportion of Classification=0 by Injection Type and Dataset:\n")
    print(table)

if __name__ == "__main__":
    main()
