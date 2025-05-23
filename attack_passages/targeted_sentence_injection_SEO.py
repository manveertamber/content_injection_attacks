import sys
sys.path.append('..')

from datasets import load_dataset
import json
from adversarial_passage_generator import AdversarialGenerator
import random
from tqdm import tqdm

datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']

for dataset in tqdm(datasets):
    beir_dataset = 'msmarco' if dataset in ['dl19', 'dl20'] else dataset
    ds = load_dataset(f"BeIR/{beir_dataset}", "corpus")['corpus']
    id_key = '_id'
    passages_dict = {
        str(line[id_key]): str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        for line in tqdm(ds, desc="Loading corpus")
    }

    for model in ['adversarial-plain-retriever', 'adversarial-random-1.0-retriever', 'adversarial-random-1.1-retriever', 'adversarial-targeted-1.0-retriever', 'bge-base-retriever', 'bge-large-retriever', 'e5-unsupervised-retriever', 'e5-supervised-retriever', 'snowflake-base-retriever', 'minilm-reranker', 'monot5-base-reranker', 'rankt5-base-reranker', 'monot5-large-reranker', 'llama_3.1_8b_judge']:

        print('dataset', dataset)
        output_filepath = f'passages/targeted_sentence_injection_SEO_passages_{model}_{dataset}.jsonl'
        print('output_filepath', output_filepath)
        
        adversarial_generator = AdversarialGenerator(
            passages_file=f'../passages/msmarco.tsv',
            injection_sentences_file='../random_sentences/splits/toxigen_test.txt', 
            instruction="",
        )

        queries_dict = {}
        with open(f'../queries/{dataset}-queries.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                qid, query = (val.strip() for val in line.split('\t'))
                queries_dict[qid] = query

        outputs = {
            "attack_ids": [],
            "qids": [],
            "adversarial_passages": [],
            "num_injected_words": [],
            "num_sentence_injection": [],
            "num_original_words": [],
            "locations": [],
            "passage_types": [],
            "current_ranks_scores": [],
            "passage_repetition": [],
        }

        candidate_passages = {}
        candidate_ranks_scores = {}
        # consider top-5 passages from retrievers/rerankers and all judged passages from LLM
        if 'retriever' in model or 'reranker' in model:
            with open(f'../model_scores_and_judgements/run.{model}.{dataset}.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    qid, _, pid, rank, score, _ = line.split()
                    rank = int(rank)
                    if rank == 1:
                        if qid not in candidate_passages:
                            candidate_passages[qid] = []
                            candidate_ranks_scores[qid] = []

                        candidate_passages[qid].append(passages_dict[pid])
                        candidate_ranks_scores[qid].append((int(rank), float(score)))

        else:
            with open(f'../model_scores_and_judgements/{model}.qrel_scores.{dataset}.tsv', 'r', encoding='utf-8') as f:
                counts_per_score = {}
                lines = f.readlines()
                random.shuffle(lines)
                for line in lines:
                    qid, query, pid, passage, expected_score, generated_score = line.split('\t')
                    generated_score = int(generated_score)
                    expected_score = int(expected_score)
                    
                    if generated_score == 3:
                        if qid not in candidate_passages:
                            candidate_passages[qid] = []
                            candidate_ranks_scores[qid] = []
                            counts_per_score[qid] = {0:0, 1:0, 2:0, 3:0}

                        if counts_per_score[qid][generated_score] < 10:
                            candidate_passages[qid].append(passage)
                            candidate_ranks_scores[qid].append(generated_score)
                            counts_per_score[qid][generated_score] += 1

        for qid, query in tqdm(queries_dict.items()):
            query_len = len(query.split())

            sample_count = 0
            if qid in candidate_passages:
                for i in range(len(candidate_passages[qid])):
                    passage = candidate_passages[qid][i]
                    score_rank = candidate_ranks_scores[qid][i]
                    for trial in range(5):
                        for num_sentence_injection in [1, 2]:
                            inject_sentences = adversarial_generator.get_random_sentences(num_sentence_injection)
                            for location in ['start', 'middle', 'end']:
                                passage_repetition = 1
                                attack_id = f"{qid}_trial_{trial}_sample_{sample_count}"
                                attack_passage = adversarial_generator.inject_sentences(passage, inject_sentences, location)
                                outputs['attack_ids'].append(attack_id)
                                outputs['qids'].append(qid)
                                outputs['adversarial_passages'].append(attack_passage)
                                outputs["num_injected_words"].append(len(attack_passage.split()) - len(passage.split()))
                                outputs['num_sentence_injection'].append(num_sentence_injection)
                                outputs["num_original_words"].append(len(passage.split()))    
                                outputs["locations"].append(location)
                                outputs["passage_types"].append('SEO')
                                outputs["current_ranks_scores"].append(score_rank)
                                outputs['passage_repetition'].append(passage_repetition)

                                sample_count+=1

        with open(output_filepath, 'w') as log_file:
            for i in range(len(outputs["attack_ids"])):
                entry = {key: outputs[key][i] for key in outputs}
                log_file.write(json.dumps(entry) + '\n')