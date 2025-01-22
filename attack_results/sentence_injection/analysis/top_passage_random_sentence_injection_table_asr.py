import json
import os
from collections import defaultdict
from tabulate import tabulate

INCLUDE_ALL_REPS = True  
INCLUDE_ALL_LOCATIONS = True

reranker_files = [
    "../random_sentences/random_sentence_injection_SEO_passages_minilm-reranker_dl19_minilm-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_monot5-base-reranker_dl19_monot5-base-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_monot5-large-reranker_dl19_monot5-large-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_rankt5-base-reranker_dl19_rankt5-base-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_minilm-reranker_dl20_minilm-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_monot5-base-reranker_dl20_monot5-base-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_monot5-large-reranker_dl20_monot5-large-reranker.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_rankt5-base-reranker_dl20_rankt5-base-reranker.jsonl"
]

retriever_files = [
    "../random_sentences/random_sentence_injection_SEO_passages_bge-base-retriever_dl19_bge-base-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_bge-large-retriever_dl19_bge-large-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_e5-supervised-retriever_dl19_e5-supervised-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_e5-unsupervised-retriever_dl19_e5-unsupervised-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_snowflake-base-retriever_dl19_snowflake-base-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_bge-base-retriever_dl20_bge-base-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_bge-large-retriever_dl20_bge-large-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_e5-supervised-retriever_dl20_e5-supervised-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_e5-unsupervised-retriever_dl20_e5-unsupervised-retriever.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_snowflake-base-retriever_dl20_snowflake-base-retriever.jsonl"
]

judge_files = [
    "../random_sentences/random_sentence_injection_SEO_passages_gpt4o_dl19_gpt4o.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_llama_3.1_8b_judge_dl19_llama_3.1_8b_judge.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_gpt4o_dl20_gpt4o.jsonl",
    "../random_sentences/random_sentence_injection_SEO_passages_llama_3.1_8b_judge_dl20_llama_3.1_8b_judge.jsonl"
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

    valid_passage_types = {"SEO"}

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
                    lambda: defaultdict(aggregator_factory)
                )
            )
        )
    )

    aggregator_retrievers = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(aggregator_factory)
                )
            )
        )
    )

    aggregator_judges = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(aggregator_factory)
                )
            )
        )
    )

    for fname in reranker_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())

                if "control" in entry.get("attack_ids", ""):
                    continue

                current_ranks_scores = entry.get("current_ranks_scores", [])
                if not current_ranks_scores:
                    continue
                if current_ranks_scores[0] != 1:
                    continue

                loc = entry.get('locations')
                p_type = entry.get('passage_types')
                num_reps = entry.get('num_sentence_injection', None)
                rank = entry.get('rank', 1001)
                passage_rep = entry.get('passage_repetition', None)

                if num_reps not in reps_to_consider:
                    continue
                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue

                agg_data = aggregator_rerankers[model_name][loc][p_type][num_reps][passage_rep]
                agg_data["total"] += 1

                if rank == 1:
                    agg_data["strict_success"] += 1

                if rank <= 5:
                    agg_data["relaxed_success"] += 1

    for fname in retriever_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())

                # Skip control records
                if "control" in entry.get("attack_ids", ""):
                    continue

                current_ranks_scores = entry.get("current_ranks_scores", [])
                if not current_ranks_scores:
                    continue
                if current_ranks_scores[0] != 1:
                    continue

                loc = entry.get('locations')
                p_type = entry.get('passage_types')
                num_reps = entry.get('num_sentence_injection', None)
                rank = entry.get('rank', 1001)
                passage_rep = entry.get('passage_repetition', None)

                if num_reps not in reps_to_consider:
                    continue
                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue

                agg_data = aggregator_retrievers[model_name][loc][p_type][num_reps][passage_rep]
                agg_data["total"] += 1

                if rank == 1:
                    agg_data["strict_success"] += 1

                if rank <= 5:
                    agg_data["relaxed_success"] += 1

    for fname in judge_files:
        model_name = extract_model_name(fname)
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())

                if "control" in entry.get("attack_ids", ""):
                    continue
                
                current_ranks_scores = entry.get("current_ranks_scores", [])
                if current_ranks_scores != 3:
                    continue

                score = entry.get("score", None)
                if score is None:
                    continue

                loc = entry.get('locations')
                p_type = entry.get('passage_types')
                num_reps = entry.get('num_sentence_injection', None)
                passage_rep = entry.get('passage_repetition', None)

                if num_reps not in reps_to_consider:
                    continue
                if loc not in valid_locations:
                    continue
                if p_type not in valid_passage_types:
                    continue

                agg_data = aggregator_judges[model_name][loc][p_type][num_reps][passage_rep]
                agg_data["total"] += 1

                if score == 3:
                    agg_data["strict_success"] += 1

                if score >= 2:
                    agg_data["relaxed_success"] += 1


    def build_asr_table(aggregator, title, is_judge=False):

        all_models = sorted(aggregator.keys())

        passage_rep_values = set()
        for model in aggregator:
            for loc_ in aggregator[model]:
                for p_type_ in aggregator[model][loc_]:
                    for reps_ in aggregator[model][loc_][p_type_]:
                        for p_rep_ in aggregator[model][loc_][p_type_][reps_]:
                            passage_rep_values.add(p_rep_)

        passage_rep_values = sorted(passage_rep_values)


        header = ["PassageType", "PassageRep", "#Reps", "Location"]
        for m in all_models:
            if not is_judge:
                header.append(f"{m}(=1)") 
                header.append(f"{m}(<=5)")
            else:
                header.append(f"{m}(=3)")
                header.append(f"{m}(>=2)")

        rows = []
        for p_type in sorted(valid_passage_types):
            for p_rep in passage_rep_values:
                for reps_ in sorted(reps_to_consider):
                    for loc_ in sorted(valid_locations):
                        row_data = [p_type, p_rep, reps_, loc_]
                        for m in all_models:
                            data = aggregator[m][loc_][p_type][reps_].get(p_rep, 
                                {"total": 0, "strict_success": 0, "relaxed_success": 0}
                            )
                            total = data["total"]
                            strict_succ = data["strict_success"]
                            relax_succ = data["relaxed_success"]
                            if total > 0:
                                asr_strict = strict_succ / total * 100
                                asr_relaxed = relax_succ / total * 100
                            else:
                                asr_strict = 0.0
                                asr_relaxed = 0.0
                            row_data.append(f"{asr_strict:.1f}")
                            row_data.append(f"{asr_relaxed:.1f}")
                        rows.append(row_data)

        print(f"\n{title}\n" + "-"*50)
        if rows:
            print(tabulate(rows, headers=header, tablefmt="latex"))
        else:
            print("No data found for the current configuration.\n")


    build_asr_table(aggregator_rerankers,  title="RERANKER ASR RESULTS", is_judge=False)
    build_asr_table(aggregator_retrievers, title="RETRIEVER ASR RESULTS", is_judge=False)
    build_asr_table(aggregator_judges,     title="LLM JUDGE ASR RESULTS", is_judge=True)


if __name__ == "__main__":
    main()
