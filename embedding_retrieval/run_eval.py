from tqdm import tqdm
import json
from embedding_model import EmbeddingModel
import numpy as np
import torch

batch_size = 1024
datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']

models = ['e5-supervised-retriever', 'bge-base-retriever', 'bge-large-retriever', 'e5-unsupervised-retriever', 'snowflake-base-retriever']
models += ['adversarial-plain-retriever', 'adversarial-random-1.0-retriever', 'adversarial-random-1.1-retriever', 'adversarial-targeted-1.0-retriever', ]

instructions = ["query: ", "Represent this sentence for searching relevant passages: ", "Represent this sentence for searching relevant passages: ", "query: ", "Represent this sentence for searching relevant passages: "]
instructions += ["Represent this sentence for searching relevant passages: ", "Represent this sentence for searching relevant passages: ", "Represent this sentence for searching relevant passages: ", "Represent this sentence for searching relevant passages: ",]

passage_prefixes = ["passage: ", "", "", "passage: ", ""]
passage_prefixes += ["","","",""]

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
    instruction = instructions[i]
    passage_prefix = passage_prefixes[i]
    hgf_model_name = ""
    
    pooling='cls'
    
    if model_name == 'bge-base-retriever':
        hgf_model_name = 'BAAI/bge-base-en-v1.5'
    elif model_name == 'bge-large-retriever':
        hgf_model_name = 'BAAI/bge-large-en-v1.5'
    elif model_name == 'e5-unsupervised-retriever':
        hgf_model_name = 'intfloat/e5-base-unsupervised'
        pooling='mean'
    elif model_name == 'e5-supervised-retriever':
        hgf_model_name = 'intfloat/e5-base'
        pooling='mean'
    elif model_name == 'snowflake-base-retriever':
        hgf_model_name = 'Snowflake/snowflake-arctic-embed-m-v1.5'
    else:
        hgf_model_name = f'models/{model_name}'
    
    emb_model = EmbeddingModel(hgf_model_name, normalize=True, pooling=pooling, instruction=instruction, text_maxlength=512, device='cuda', passage_prefix=passage_prefix).eval()
    
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
            retrieval_results_scores[qid].reverse()
        
        for eval_passage_file in eval_passages_files:
            output_path = '../attack_results/' + eval_passage_file.split('/')[-1].replace('.jsonl', '') + f'_{model_name}.jsonl'
            
            queries = []
            passages = []
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                for line in input_f:
                    example = json.loads(line)
                    queries.append(queries_dict[example['qids']])
                    passages.append(example['adversarial_passages'])
            
            scores = []
            for i in tqdm(range(0, len(queries), batch_size)):
                with torch.no_grad():
                    query_embeddings = emb_model.encode_queries(queries[i:i+batch_size]).cpu().detach().numpy()
                    passage_embeddings = emb_model.encode_texts(passages[i:i+batch_size]).cpu().detach().numpy()
                    scores.extend((query_embeddings * passage_embeddings).sum(axis=1).tolist())

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
