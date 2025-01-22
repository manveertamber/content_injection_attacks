from tqdm import tqdm
import json
from reranker_model import RerankerModel
import numpy as np
import torch

batch_size = 256
datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']
models = ['minilm-reranker', 'monot5-base-reranker', 'rankt5-base-reranker', 'monot5-large-reranker', ]

def binary_search(target, ordered_list):
    low, high = 0, len(ordered_list) - 1
    while low <= high:
        mid = (low + high) // 2
        if ordered_list[mid] < target:
            low = mid + 1
        elif ordered_list[mid] > target:
            high = mid - 1
        else:
            return mid
    return low

for i in range(len(models)):
    model_name = models[i]
    hgf_model_name = ""
    if model_name == 'minilm-reranker':
        hgf_model_name = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    elif model_name == 'monot5-base-reranker':
        hgf_model_name = 'castorini/monot5-base-msmarco-10k'
    elif model_name == 'rankt5-base-reranker':
        hgf_model_name = 'Soyoung97/RankT5-base'
    elif model_name == 'monot5-large-reranker':
        hgf_model_name = 'castorini/monot5-large-msmarco-10k'
    else:
        hgf_model_name = f'models/{model_name}'
    reranker_model = RerankerModel(hgf_model_name, text_maxlength=512, batch_size=batch_size, device='cuda').eval()
    
    for dataset in datasets:
        print("dataset", dataset)
        eval_passages_files = [f'../attack_passages/passages/keyword_injection_general_passages_{dataset}.jsonl',
                               f'../attack_passages/passages/keyword_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                               f'../attack_passages/passages/query_injection_general_passages_{dataset}.jsonl',
                               f'../attack_passages/passages/query_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                               f'../attack_passages/passages/random_sentence_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                               f'../attack_passages/passages/targeted_sentence_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                               ]

        if dataset in ['dl19', 'dl20']:
            eval_passages_files += [f'../attack_passages/passages/generated_random_sentence_injection_{dataset}.jsonl', 
                                    f'../attack_passages/passages/generated_targeted_sentence_injection_{dataset}.jsonl']
        
        
        queries_dict = {}
        with open(f'../queries/{dataset}-queries.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                vals = line.split('\t')
                queries_dict[vals[0].strip()] =  vals[1].strip()
       
        run_file_path = f"../model_scores_and_judgements/run.{model_name}.{dataset}.txt"

        retrieval_results_scores = {}
        with open(run_file_path, 'r') as f:
            for line in f:
                vals = line.split(' ')
                qid = vals[0]
                score = vals[4]
                if int(vals[3]) == 1:
                    retrieval_results_scores[str(qid)] = []
                retrieval_results_scores[str(qid)].append(float(score))

        for qid in retrieval_results_scores.keys():
            retrieval_results_scores[qid].reverse() # for binary search
        
        for eval_passage_file in eval_passages_files:
            output_path = '../attack_results/' + eval_passage_file.split('/')[-1].replace('.jsonl', '') + f'_{model_name}.jsonl'
            
            queries = []
            passages = []
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                for line in input_f:
                    example = json.loads(line)
                    queries.append(queries_dict[example['qids']])
                    passages.append(example['adversarial_passages'])
            
            pairs = []
            for i in range(len(queries)):
                pairs.append([queries[i], passages[i]])
           
            scores = reranker_model.score_pairs(pairs)

            assert(len(scores) == len(queries))
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                with open(output_path, 'w', encoding='utf-8') as output_f:
                    index = -1
                    for line in input_f:
                        index += 1
                        example = json.loads(line)
                        qid = example['qids']
                        if qid in retrieval_results_scores:
                            example.pop('adversarial_passages')
                            example['rank'] = len(retrieval_results_scores[qid]) + 1 - binary_search(scores[index], retrieval_results_scores[qid])
                            example['score'] = scores[index]
                            output_f.write(json.dumps(example, ensure_ascii=False) + '\n')
