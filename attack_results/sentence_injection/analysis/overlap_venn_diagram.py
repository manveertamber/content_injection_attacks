import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from tabulate import tabulate

valid_passage_types = {"gen_pos_passage"}
MODELS_OF_INTEREST = {"gpt4o", "monot5-large", "bge-large"}

reranker_files = [
    "../random_sentences/generated_random_sentence_injection_dl19_minilm-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_monot5-base-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_monot5-large-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_rankt5-base-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_minilm-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_monot5-base-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_monot5-large-reranker.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_rankt5-base-reranker.jsonl"
]

retriever_files = [
    "../random_sentences/generated_random_sentence_injection_dl19_bge-base-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_bge-large-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_e5-supervised-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_e5-unsupervised-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_snowflake-base-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_bge-base-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_bge-large-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_e5-supervised-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_e5-unsupervised-retriever.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_snowflake-base-retriever.jsonl"
]

judge_files = [
    "../random_sentences/generated_random_sentence_injection_dl19_gpt4o.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl19_llama_3.1_8b_judge.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_gpt4o.jsonl",
    "../random_sentences/generated_random_sentence_injection_dl20_llama_3.1_8b_judge.jsonl"
]

def extract_model_name(fname):
    base_name = os.path.basename(fname)
    parts = base_name.split('_')

    if "dl19" in parts or "dl20" in parts:
        dl_index = parts.index("dl19") if "dl19" in parts else parts.index("dl20")
        model_name = '_'.join(parts[dl_index + 1:]).replace(".jsonl", "")
        return model_name
    return base_name.replace(".jsonl", "")

def main():
    aggregator_rerankers = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"total": 0, "success": 0})
            )
        )
    )
    aggregator_retrievers = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"total": 0, "success": 0})
            )
        )
    )
    aggregator_judges = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"total": 0, "success": 0})
            )
        )
    )

    success_sets = {
        "gpt4o": set(),
        "monot5-large": set(),
        "bge-large": set(),
    }

    def track_success(model_name, attack_id):
        if "gpt4o" in model_name:
            success_sets["gpt4o"].add(attack_id)
        elif "monot5-large" in model_name:
            success_sets["monot5-large"].add(attack_id)
        elif "bge-large" in model_name:
            success_sets["bge-large"].add(attack_id)


    for fname in reranker_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if "control" in entry.get("attack_ids", ""):
                    continue

                p_type = entry.get('passage_types', "unknown")
                rank = entry.get('rank', 1001)
                attack_id = entry.get('attack_ids', None)
                num_reps = entry.get('num_sentence_injection', None)
                loc = entry.get('locations')

                if p_type not in valid_passage_types:
                    continue

                aggregator_rerankers[model_name][p_type][num_reps][loc]["total"] += 1

                if rank == 1:
                    aggregator_rerankers[model_name][p_type][num_reps][loc]["success"] += 1
                    if attack_id:
                        track_success(model_name, attack_id)

    for fname in retriever_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if "control" in entry.get("attack_ids", ""):
                    continue

                p_type = entry.get('passage_types', "unknown")
                rank = entry.get('rank', 1001)
                attack_id = entry.get('attack_ids', None)
                num_reps = entry.get('num_sentence_injection', None)
                loc = entry.get('locations')

                if p_type not in valid_passage_types:
                    continue

                aggregator_retrievers[model_name][p_type][num_reps][loc]["total"] += 1

                if rank == 1:
                    aggregator_retrievers[model_name][p_type][num_reps][loc]["success"] += 1
                    if attack_id:
                        track_success(model_name, attack_id)

    for fname in judge_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if "control" in entry.get("attack_ids", ""):
                    continue

                p_type = entry.get('passage_types', "unknown")
                score = entry.get("score", None)
                attack_id = entry.get('attack_ids', None)
                num_reps = entry.get('num_sentence_injection', None)
                loc = entry.get('locations')

                if p_type not in valid_passage_types:
                    continue

                aggregator_judges[model_name][p_type][num_reps][loc]["total"] += 1

                if score == 3:
                    aggregator_judges[model_name][p_type][num_reps][loc]["success"] += 1
                    if attack_id:
                        track_success(model_name, attack_id)

    gpt4o_set = success_sets["gpt4o"]
    monot5_large_set = success_sets["monot5-large"]
    bge_large_set = success_sets["bge-large"]

    print("\n=== Venn Diagram Data (no location/reps filtering) ===")
    print(f"GPT4o success set size:         {len(gpt4o_set)}")
    print(f"monot5-large success set size:  {len(monot5_large_set)}")
    print(f"bge-large success set size:     {len(bge_large_set)}")

    if gpt4o_set or monot5_large_set or bge_large_set:
        plt.figure(figsize=(8, 8))

        venn_out = venn3(
            subsets=[
                len(gpt4o_set - monot5_large_set - bge_large_set),         
                len(monot5_large_set - gpt4o_set - bge_large_set),         
                len(gpt4o_set & monot5_large_set - bge_large_set),         
                len(bge_large_set - gpt4o_set - monot5_large_set),         
                len(gpt4o_set & bge_large_set - monot5_large_set),         
                len(monot5_large_set & bge_large_set - gpt4o_set),         
                len(gpt4o_set & monot5_large_set & bge_large_set)          
            ],
            set_labels=("GPT4o", "MonoT5-large", "BGE-large"),
            set_colors=("blue", "red", "green")
        )

        for text in venn_out.set_labels:
            if text:
                text.set_fontsize(20)

        for text in venn_out.subset_labels:
            if text:
                text.set_fontsize(19)

        outfname = "sentence_injection_venn_diagram.png"
        plt.savefig(outfname, dpi=1000, bbox_inches="tight")
    else:
        print("All three sets are empty. Skipping Venn diagram.")

if __name__ == "__main__":
    main()
