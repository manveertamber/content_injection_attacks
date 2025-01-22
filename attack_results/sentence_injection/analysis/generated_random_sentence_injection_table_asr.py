import json
import os
from collections import defaultdict
from tabulate import tabulate

INCLUDE_ALL_REPS = True  
INCLUDE_ALL_LOCATIONS = True

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
    if INCLUDE_ALL_REPS:
        reps_to_consider = [1, 2]
    else:
        reps_to_consider = [1]

    if INCLUDE_ALL_LOCATIONS:
        valid_locations = {"start", "middle", "end"}
    else:
        valid_locations = {"start"}

    valid_passage_types = {"gen_pos_passage"}

    valid_passage_lengths = {50, 100, 200}

    def aggregator_factory():
        return {
            "total": 0,
            "strict_success": 0,
            "relaxed_success": 0
        }

    aggregator_rerankers = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(aggregator_factory)
                    )
                )
            )
        )
    )

    aggregator_retrievers = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(aggregator_factory)
                    )
                )
            )
        )
    )

    aggregator_judges = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(aggregator_factory)
                    )
                )
            )
        )
    )

    for fname in reranker_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                p_type = entry.get("passage_types")
                num_reps = entry.get("num_sentence_injection")
                passage_rep = entry.get("passage_repetition", None)
                p_len = entry.get("passage_length", None)
                rank = entry.get("rank", 1001)

                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue
                if num_reps not in reps_to_consider:
                    continue
                if p_len not in valid_passage_lengths:
                    continue

                agg_data = aggregator_rerankers[model_name][loc][p_type][num_reps][passage_rep][p_len]
                agg_data["total"] += 1

                if rank == 1:
                    agg_data["strict_success"] += 1
                if rank <= 5:
                    agg_data["relaxed_success"] += 1

    for fname in retriever_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Skip control records
                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                p_type = entry.get("passage_types")
                num_reps = entry.get("num_sentence_injection")
                passage_rep = entry.get("passage_repetition", None)
                p_len = entry.get("passage_length", None)
                rank = entry.get("rank", 1001)

                # Filter
                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue
                if num_reps not in reps_to_consider:
                    continue
                if p_len not in valid_passage_lengths:
                    continue

                agg_data = aggregator_retrievers[model_name][loc][p_type][num_reps][passage_rep][p_len]
                agg_data["total"] += 1

                if rank == 1:
                    agg_data["strict_success"] += 1
                if rank <= 5:
                    agg_data["relaxed_success"] += 1

    for fname in judge_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Skip control records
                if "control" in entry.get("attack_ids", ""):
                    continue

                loc = entry.get("locations")
                p_type = entry.get("passage_types")
                num_reps = entry.get("num_sentence_injection")
                passage_rep = entry.get("passage_repetition", None)
                p_len = entry.get("passage_length", None)
                score = entry.get("score", None)

                # Filter
                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue
                if num_reps not in reps_to_consider:
                    continue
                if p_len not in valid_passage_lengths:
                    continue
                if score is None:
                    continue

                agg_data = aggregator_judges[model_name][loc][p_type][num_reps][passage_rep][p_len]
                agg_data["total"] += 1

                if score == 3:
                    agg_data["strict_success"] += 1
                if score >= 2:
                    agg_data["relaxed_success"] += 1

    def build_asr_table(aggregator, title, is_judge=False):

        all_models = sorted(aggregator.keys())

        passage_rep_values = set()
        passage_length_values = set()

        for m in aggregator:
            for loc_ in aggregator[m]:
                for pt_ in aggregator[m][loc_]:
                    for reps_ in aggregator[m][loc_][pt_]:
                        for prep_ in aggregator[m][loc_][pt_][reps_]:
                            for plen_ in aggregator[m][loc_][pt_][reps_][prep_]:
                                passage_rep_values.add(prep_)
                                passage_length_values.add(plen_)

        passage_rep_values = sorted(passage_rep_values)
        passage_length_values = sorted(passage_length_values)

        header = ["PassageType", "PassageRep", "#Reps", "Location", "PassageLength"]
        for m in all_models:
            if not is_judge:
                header.append(f"{m}(Top-1)")
                header.append(f"{m}(Top-5)")
            else:
                header.append(f"{m}(Score=3)")
                header.append(f"{m}(Score≥2)")

        rows = []
        for pt_ in sorted(valid_passage_types):
            for prep_ in passage_rep_values:
                for reps_ in sorted(reps_to_consider):
                    for loc_ in sorted(valid_locations):
                        for plen_ in passage_length_values:
                            row_data = [pt_, prep_, reps_, loc_, plen_]
                            row_has_any_data = False
                            for m in all_models:
                                data = aggregator[m][loc_][pt_][reps_][prep_].get(plen_, 
                                    {"total":0, "strict_success":0, "relaxed_success":0}
                                )
                                total = data["total"]
                                strict_succ = data["strict_success"]
                                relax_succ = data["relaxed_success"]

                                if total > 0:
                                    row_has_any_data = True
                                    asr_strict = strict_succ / total * 100
                                    asr_relaxed = relax_succ / total * 100
                                else:
                                    asr_strict = 0.0
                                    asr_relaxed = 0.0

                                row_data.append(f"{asr_strict:.1f}")
                                row_data.append(f"{asr_relaxed:.1f}")
                            
                            if row_has_any_data:
                                rows.append(row_data)

        print(f"\n{title}\n" + "-"*60)
        if rows:
            print(tabulate(rows, headers=header, tablefmt="github"))
        else:
            print("No data found for the current configuration.\n")

    build_asr_table(aggregator_rerankers,  title="RERANKER ASR RESULTS", is_judge=False)
    build_asr_table(aggregator_retrievers, title="RETRIEVER ASR RESULTS", is_judge=False)
    build_asr_table(aggregator_judges,     title="LLM JUDGE ASR RESULTS", is_judge=True)


if __name__ == "__main__":
    main()
