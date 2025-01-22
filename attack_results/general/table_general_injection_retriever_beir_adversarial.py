import json
import os
from collections import defaultdict
from tabulate import tabulate

ALLOWED_RETRIEVERS = {
    "bge-base-retriever",
    "adversarial-plain-retriever",
    "adversarial-random-1.1-retriever",
    "adversarial-random-1.0-retriever",
    "adversarial-targeted-1.0-retriever",
}

beir_datasets = [
    "dl19",
    "dl20",
    "fiqa",
    "scifact",
    "trec-covid",
    "nfcorpus",
    "dbpedia-entity",
    "climate-fever",
]

def get_dataset_row_order():
    special_first = ["dl19", "dl20"]
    remainder = sorted(set(beir_datasets) - set(special_first))
    return special_first + remainder


def model_is_allowed(model_name: str, allowed_set) -> bool:
    m_lower = model_name.lower()
    return any(token in m_lower for token in allowed_set)


query_injection_retriever_files = []
for ds in beir_datasets:
    for retr_model in ALLOWED_RETRIEVERS:
        path = f"../query_injection/query_injection_general_passages_{ds}_{retr_model}.jsonl"
        query_injection_retriever_files.append(path)

keyword_injection_retriever_files = []
for ds in beir_datasets:
    for retr_model in ALLOWED_RETRIEVERS:
        path = f"../keyword_injection/keyword_injection_general_passages_{ds}_{retr_model}.jsonl"
        keyword_injection_retriever_files.append(path)

msmarco_sentence_injection_retriever_files = []
for ds in beir_datasets:
    for retr_model in ALLOWED_RETRIEVERS:
        path = f"../sentence_injection/random_sentences/random_sentence_injection_SEO_passages_{retr_model}_{ds}_{retr_model}.jsonl"
        msmarco_sentence_injection_retriever_files.append(path)

toxigen_sentence_injection_retriever_files = []
for ds in beir_datasets:
    for retr_model in ALLOWED_RETRIEVERS:
        path = f"../sentence_injection/targeted_sentences/targeted_sentence_injection_SEO_passages_{retr_model}_{ds}_{retr_model}.jsonl"
        toxigen_sentence_injection_retriever_files.append(path)

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

    aggregator_qi_retrievers = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
    )

    for fname in query_injection_retriever_files:
        if not os.path.exists(fname):
            continue
        ds, m = parse_dataset_and_model_qi(fname)
        if not model_is_allowed(m, ALLOWED_RETRIEVERS):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue

                rank = entry.get("rank", 1001)
                aggregator_qi_retrievers[ds][m]["total"] += 1
                if rank == 1:
                    aggregator_qi_retrievers[ds][m]["top1"] += 1
                if rank <= 5:
                    aggregator_qi_retrievers[ds][m]["top5"] += 1

    def print_table_qi_aggregated(aggregator, title):
        ds_rows = get_dataset_row_order()
        all_models = set()
        for ds in aggregator:
            for mo in aggregator[ds]:
                all_models.add(mo)
        all_models = sorted(all_models)

        table_rows = []
        for ds in ds_rows:
            row = [ds]
            for m in all_models:
                stats = aggregator[ds][m] if m in aggregator[ds] else {"total": 0, "top1": 0, "top5": 0}
                total = stats["total"]
                asr_top1 = (stats["top1"] / total * 100) if total > 0 else 0.0
                asr_top5 = (stats["top5"] / total * 100) if total > 0 else 0.0
                row.extend([f"{asr_top1:.1f}", f"{asr_top5:.1f}"])
            table_rows.append(row)

        headers = ["Dataset"]
        for m in all_models:
            headers.extend([f"{m} Top1", f"{m} Top5"])
        print(f"\n{title} (Aggregated over Passage Type, Location, etc.)\n{'-'*60}")
        if table_rows:
            print(tabulate(table_rows, headers=headers, tablefmt="latex"))
        else:
            print("No data found.")

    print_table_qi_aggregated(aggregator_qi_retrievers, "QUERY INJECTION: RETRIEVERS")

    aggregator_ki_retrievers = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
    )

    valid_locs_ki = {"start", "middle", "end"}

    for fname in keyword_injection_retriever_files:
        if not os.path.exists(fname):
            continue
        ds_name, model_name = parse_dataset_and_model_ki(fname)
        if not model_is_allowed(model_name, ALLOWED_RETRIEVERS):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                rank = entry.get("rank", 1001)

                if loc not in valid_locs_ki:
                    continue

                aggregator_ki_retrievers[ds_name][model_name]["total"] += 1
                if rank == 1:
                    aggregator_ki_retrievers[ds_name][model_name]["top1"] += 1
                if rank <= 5:
                    aggregator_ki_retrievers[ds_name][model_name]["top5"] += 1

    def print_table_keyword_injection_aggregated(aggregator, title):
        ds_rows = get_dataset_row_order()
        all_models = set()
        for ds in aggregator:
            for mo in aggregator[ds]:
                all_models.add(mo)
        all_models = sorted(all_models)

        table_rows = []
        for ds in ds_rows:
            row = [ds]
            for m in all_models:
                stats = aggregator[ds][m] if m in aggregator[ds] else {"total": 0, "top1": 0, "top5": 0}
                total = stats["total"]
                asr_top1 = (stats["top1"]/ total * 100) if total > 0 else 0.0
                asr_top5 = (stats["top5"]/ total * 100) if total > 0 else 0.0
                row.extend([f"{asr_top1:.1f}", f"{asr_top5:.1f}"])
            table_rows.append(row)

        headers = ["Dataset"]
        for m in all_models:
            headers.extend([f"{m} Top1", f"{m} Top5"])
        print(f"\n{title} (Aggregated over Location & Expand)\n{'-'*60}")
        if table_rows:
            print(tabulate(table_rows, headers=headers, tablefmt="latex"))
        else:
            print("No data found.")

    print_table_keyword_injection_aggregated(aggregator_ki_retrievers, "KEYWORD INJECTION: RETRIEVERS")

    aggregator_msmarco_si_retrievers = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})
    aggregator_toxigen_si_retrievers = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})

    valid_locs_si = {"start", "middle", "end"}

    for fname in msmarco_sentence_injection_retriever_files:
        if not os.path.exists(fname):
            continue
        model_str = extract_sentence_model_name(fname)
        if not model_is_allowed(model_str, ALLOWED_RETRIEVERS):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                rank = entry.get("rank", 1001)

                if loc not in valid_locs_si:
                    continue

                aggregator_msmarco_si_retrievers[model_str]["total"] += 1
                if rank == 1:
                    aggregator_msmarco_si_retrievers[model_str]["top1"] += 1
                if rank <= 5:
                    aggregator_msmarco_si_retrievers[model_str]["top5"] += 1

    for fname in toxigen_sentence_injection_retriever_files:
        if not os.path.exists(fname):
            continue
        model_str = extract_sentence_model_name(fname)
        if not model_is_allowed(model_str, ALLOWED_RETRIEVERS):
            continue

        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "control" in entry.get("attack_ids", ""):
                    continue
                loc = entry.get("locations")
                rank = entry.get("rank", 1001)

                if loc not in valid_locs_si:
                    continue

                aggregator_toxigen_si_retrievers[model_str]["total"] += 1
                if rank == 1:
                    aggregator_toxigen_si_retrievers[model_str]["top1"] += 1
                if rank <= 5:
                    aggregator_toxigen_si_retrievers[model_str]["top5"] += 1

    def print_table_sentence_injection_aggregated(aggregator, title):
        all_models = sorted(aggregator.keys())
        header = ["Model", "ASR Top1 (%)", "ASR Top5 (%)"]
        rows = []
        for model in all_models:
            stats = aggregator[model]
            total = stats["total"]
            asr_top1 = (stats["top1"] / total * 100) if total > 0 else 0.0
            asr_top5 = (stats["top5"] / total * 100) if total > 0 else 0.0
            rows.append([model, f"{asr_top1:.1f}", f"{asr_top5:.1f}"])

        print(f"\n{title} (Aggregated over Location, #Reps, etc.)\n{'-'*60}")
        if rows:
            print(tabulate(rows, headers=header, tablefmt="latex"))
        else:
            print("No data found.")

    print_table_sentence_injection_aggregated(aggregator_msmarco_si_retrievers, 
                                              "SENTENCE INJECTION (MSMARCO): RETRIEVERS")
    print_table_sentence_injection_aggregated(aggregator_toxigen_si_retrievers, 
                                              "SENTENCE INJECTION (TOXIGEN): RETRIEVERS")

if __name__ == "__main__":
    main()
