import json
import os
from collections import defaultdict
from tabulate import tabulate


ALLOWED_JUDGES = {
    "gpt4o",
    "gpt4o-adv",
}

target_datasets = ["dl19", "dl20"]

def model_is_allowed(model_name: str, allowed_set) -> bool:
    m_lower = model_name.lower()
    return any(token in m_lower for token in allowed_set)


query_injection_judge_files = []
for ds in target_datasets:
    for judge_model in ALLOWED_JUDGES:
        path = f"../query_injection/query_injection_general_passages_{ds}_{judge_model}.jsonl"
        query_injection_judge_files.append(path)

keyword_injection_judge_files = []
for ds in target_datasets:
    for judge_model in ALLOWED_JUDGES:
        path = f"../keyword_injection/keyword_injection_general_passages_{ds}_{judge_model}.jsonl"
        keyword_injection_judge_files.append(path)

msmarco_sentence_injection_judge_files = []
for ds in target_datasets:
    for judge_model in ALLOWED_JUDGES:
        path = f"../sentence_injection/random_sentences/random_sentence_injection_SEO_passages_{judge_model}_{ds}_{judge_model}.jsonl"
        msmarco_sentence_injection_judge_files.append(path)

toxigen_sentence_injection_judge_files = []
for ds in target_datasets:
    for judge_model in ALLOWED_JUDGES:
        path = f"../sentence_injection/targeted_sentences/targeted_sentence_injection_SEO_passages_{judge_model}_{ds}_{judge_model}.jsonl"
        toxigen_sentence_injection_judge_files.append(path)


def parse_dataset_and_model_qi(fname: str):
    base = os.path.basename(fname)
    prefix = "query_injection_general_passages_"
    if base.startswith(prefix):
        base = base[len(prefix):]
    if base.endswith(".jsonl"):
        base = base[:-6]
    parts = base.split("_", maxsplit=1)
    if len(parts) == 2:
        dataset_name, model_name = parts
    else:
        dataset_name, model_name = "unknown", base
    return dataset_name, model_name

def parse_dataset_and_model_ki(fname: str):
    base = os.path.basename(fname)
    base = base.replace("keyword_injection_general_passages_", "")
    base = base.replace(".jsonl", "")
    parts = base.split("_", maxsplit=1)
    if len(parts) == 2:
        dataset_name, model_name = parts
    else:
        dataset_name, model_name = "unknown", base
    return dataset_name, model_name

def extract_sentence_model_name(fname):
    base = os.path.basename(fname)
    if base.endswith(".jsonl"):
        base = base[:-6]
    if "passages_" in base:
        parts = base.split("passages_", maxsplit=1)
        candidate = parts[-1]
        return candidate
    return base


def main():
    # 1) Query Injection
    aggregator_qi_judges = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "score3": 0, "score2plus": 0})
    )

    for fname in query_injection_judge_files:
        if not os.path.exists(fname):
            continue
        ds, m = parse_dataset_and_model_qi(fname)
        if ds not in target_datasets:
            continue
        if not model_is_allowed(m, ALLOWED_JUDGES):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue

                score = entry.get("score", None)
                aggregator_qi_judges[ds][m]["total"] += 1
                if score == 3:
                    aggregator_qi_judges[ds][m]["score3"] += 1
                if score is not None and score >= 2:
                    aggregator_qi_judges[ds][m]["score2plus"] += 1

    # 2) Keyword Injection
    aggregator_ki_judges = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "score3": 0, "score2plus": 0})
    )

    valid_locs_ki = {"start", "middle", "end"}

    for fname in keyword_injection_judge_files:
        if not os.path.exists(fname):
            continue
        ds_name, model_name = parse_dataset_and_model_ki(fname)
        if ds_name not in target_datasets:
            continue
        if not model_is_allowed(model_name, ALLOWED_JUDGES):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                score = entry.get("score", None)

                if loc not in valid_locs_ki:
                    continue

                aggregator_ki_judges[ds_name][model_name]["total"] += 1
                if score == 3:
                    aggregator_ki_judges[ds_name][model_name]["score3"] += 1
                if score is not None and score >= 2:
                    aggregator_ki_judges[ds_name][model_name]["score2plus"] += 1

    # 3) Sentence Injection (MSMARCO)
    aggregator_msmarco_si_judges = defaultdict(
        lambda: {"total": 0, "score3": 0, "score2plus": 0}
    )

    valid_locs_si = {"start", "middle", "end"}

    for fname in msmarco_sentence_injection_judge_files:
        if not os.path.exists(fname):
            continue
        model_str = extract_sentence_model_name(fname)
        if not model_is_allowed(model_str, ALLOWED_JUDGES):
            continue

        if not any(ds in model_str for ds in target_datasets):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                score = entry.get("score", None)

                if loc not in valid_locs_si:
                    continue

                aggregator_msmarco_si_judges[model_str]["total"] += 1
                if score == 3:
                    aggregator_msmarco_si_judges[model_str]["score3"] += 1
                if score is not None and score >= 2:
                    aggregator_msmarco_si_judges[model_str]["score2plus"] += 1

    # 4) Sentence Injection (Toxigen)
    aggregator_toxigen_si_judges = defaultdict(
        lambda: {"total": 0, "score3": 0, "score2plus": 0}
    )

    for fname in toxigen_sentence_injection_judge_files:
        if not os.path.exists(fname):
            continue
        model_str = extract_sentence_model_name(fname)
        if not model_is_allowed(model_str, ALLOWED_JUDGES):
            continue

        if not any(ds in model_str for ds in target_datasets):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                score = entry.get("score", None)

                if loc not in valid_locs_si:
                    continue

                aggregator_toxigen_si_judges[model_str]["total"] += 1
                if score == 3:
                    aggregator_toxigen_si_judges[model_str]["score3"] += 1
                if score is not None and score >= 2:
                    aggregator_toxigen_si_judges[model_str]["score2plus"] += 1

    def print_table_judges_aggregated(aggregator, title):
        ds_rows = target_datasets  # just dl19, dl20
        all_models = set()
        for ds in aggregator:
            for mo in aggregator[ds]:
                all_models.add(mo)
        all_models = sorted(all_models)

        table_rows = []
        for ds in ds_rows:
            row = [ds]
            for m in all_models:
                stats = aggregator[ds][m] if m in aggregator[ds] else {"total": 0, "score3": 0, "score2plus": 0}
                total = stats["total"]
                asr_score3 = (stats["score3"] / total * 100) if total > 0 else 0.0
                asr_score2plus = (stats["score2plus"] / total * 100) if total > 0 else 0.0
                row.extend([f"{asr_score3:.1f}", f"{asr_score2plus:.1f}"])
            table_rows.append(row)

        headers = ["Dataset"]
        for m in all_models:
            headers.extend([f"{m} s3", f"{m} s2+"])
        print(f"\n{title}\n{'-'*60}")
        if table_rows:
            print(tabulate(table_rows, headers=headers, tablefmt="latex"))
        else:
            print("No data found.")

    def print_table_judges_sent_injection(aggregator, title):
        """
        For sentence injection, aggregator key = model_str
        We'll keep a row only if 'dl19' or 'dl20' is found in model_str.
        Output: Model, #Total, ASR Score3 (%), ASR Score2+ (%).
        """
        # Filter for those with 'dl19' or 'dl20' in the key
        final_keys = [k for k in aggregator.keys() if any(ds in k for ds in target_datasets)]
        final_keys = sorted(final_keys)

        header = ["Model", "ASR Score3 (%)", "ASR Score2+ (%)", "Total"]
        rows = []
        for model in final_keys:
            stats = aggregator[model]
            total = stats["total"]
            asr_s3 = (stats["score3"] / total * 100) if total > 0 else 0.0
            asr_s2plus = (stats["score2plus"] / total * 100) if total > 0 else 0.0
            rows.append([model, f"{asr_s3:.1f}", f"{asr_s2plus:.1f}", total])

        print(f"\n{title}\n{'-'*60}")
        if rows:
            print(tabulate(rows, headers=header, tablefmt="latex"))
        else:
            print("No data found.")


    print_table_judges_aggregated(aggregator_qi_judges, "QUERY INJECTION: LLM JUDGES (dl19 & dl20)")

    print_table_judges_aggregated(aggregator_ki_judges, "KEYWORD INJECTION: LLM JUDGES (dl19 & dl20)")

    print_table_judges_sent_injection(
        aggregator_msmarco_si_judges, 
        "SENTENCE INJECTION (MSMARCO): LLM JUDGES (dl19 & dl20)"
    )

    print_table_judges_sent_injection(
        aggregator_toxigen_si_judges,
        "SENTENCE INJECTION (TOXIGEN): LLM JUDGES (dl19 & dl20)"
    )


if __name__ == "__main__":
    main()
