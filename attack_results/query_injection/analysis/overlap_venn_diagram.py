import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from tabulate import tabulate

valid_locations = {"start", "middle", "end"}
reps_to_consider = [1, 2, 3]

reranker_files = [
    "../query_injection_general_passages_dl19_minilm-reranker.jsonl",
    "../query_injection_general_passages_dl19_monot5-base-reranker.jsonl",
    "../query_injection_general_passages_dl19_monot5-large-reranker.jsonl",
    "../query_injection_general_passages_dl19_rankt5-base-reranker.jsonl",
    "../query_injection_general_passages_dl20_minilm-reranker.jsonl",
    "../query_injection_general_passages_dl20_monot5-base-reranker.jsonl",
    "../query_injection_general_passages_dl20_monot5-large-reranker.jsonl",
    "../query_injection_general_passages_dl20_rankt5-base-reranker.jsonl"
]

retriever_files = [
    "../query_injection_general_passages_dl19_bge-base-retriever.jsonl",
    "../query_injection_general_passages_dl19_bge-large-retriever.jsonl",
    "../query_injection_general_passages_dl19_e5-supervised-retriever.jsonl",
    "../query_injection_general_passages_dl19_e5-unsupervised-retriever.jsonl",
    "../query_injection_general_passages_dl19_snowflake-base-retriever.jsonl",
    "../query_injection_general_passages_dl20_bge-base-retriever.jsonl",
    "../query_injection_general_passages_dl20_bge-large-retriever.jsonl",
    "../query_injection_general_passages_dl20_e5-supervised-retriever.jsonl",
    "../query_injection_general_passages_dl20_e5-unsupervised-retriever.jsonl",
    "../query_injection_general_passages_dl20_snowflake-base-retriever.jsonl"
]

judge_files = [
    "../query_injection_general_passages_dl19_gpt4o.jsonl",
    "../query_injection_general_passages_dl19_llama_3.1_8b_judge.jsonl",
    "../query_injection_general_passages_dl20_gpt4o.jsonl",
    "../query_injection_general_passages_dl20_llama_3.1_8b_judge.jsonl",
]

def extract_model_name(fname):
    base_name = os.path.basename(fname)
    name = (
        base_name
        .replace("query_injection_general_passages_", "")
        .replace(".jsonl", "")
        .replace("dl19_", "")
        .replace("dl20_", "")
    )
    return name

def main():
    success_sets = {
        "MonoT5-large":   set(),
        "BGE-large":      set(),
        "GPT4o":          set(),
    }

    def track_success(model_name, attack_id):
        lower = model_name.lower()
        if "monot5-large" in lower:
            success_sets["MonoT5-large"].add(attack_id)
        elif "bge-large" in lower:
            success_sets["BGE-large"].add(attack_id)
        elif "gpt4o" in lower:
            success_sets["GPT4o"].add(attack_id)

    for fname in reranker_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())

                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                reps = entry.get("num_query_repetitions", None)
                rank = entry.get("rank", 1001)
                attack_id = entry.get("attack_ids", None)

                if loc in valid_locations and reps in reps_to_consider:
                    if rank is not None and rank == 1:
                        if attack_id:
                            track_success(model_name, attack_id)

    for fname in retriever_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())

                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                reps = entry.get("num_query_repetitions", None)
                rank = entry.get("rank", 1001)
                attack_id = entry.get("attack_ids", None)

                if loc in valid_locations and reps in reps_to_consider:
                    if rank is not None and rank == 1:
                        if attack_id:
                            track_success(model_name, attack_id)

    for fname in judge_files:
        model_name = extract_model_name(fname)
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())

                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                reps = entry.get("num_query_repetitions", None)
                score = entry.get("score", None)
                attack_id = entry.get("attack_ids", None)

                if loc in valid_locations and reps in reps_to_consider:
                    if score == 3:
                        if attack_id:
                            track_success(model_name, attack_id)


    set1_label = "MonoT5-large"
    set2_label = "BGE-large"
    set3_label = "GPT4o"

    set1 = success_sets[set1_label]
    set2 = success_sets[set2_label]
    set3 = success_sets[set3_label]

    if not (set1 or set2 or set3):
        print("\nAll three chosen sets are empty. Skipping 3-set Venn diagram.\n")
    else:
        plt.figure(figsize=(8, 6))

        venn_diagram = venn3(
            subsets=[set1, set2, set3],
            set_labels=(set1_label, set2_label, set3_label),
            set_colors=("red", "green", "blue")
        )

        for idx, subset in enumerate(venn_diagram.subset_labels):
            print(subset)

        for text in venn_diagram.set_labels:
            text.set_fontsize(20)

        for text in venn_diagram.subset_labels:
            text.set_fontsize(19)

        outfname = "query_injection_venn_diagram.png"
        plt.savefig(outfname, dpi=1000, bbox_inches="tight")
  
    for model_label, sset in success_sets.items():
        print(f"{model_label:<15} => {len(sset)}")

if __name__ == "__main__":
    main()
