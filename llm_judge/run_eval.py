from tqdm import tqdm
import json
from llm_judge import LLMJudge
import numpy as np
import torch

batch_size = 256
datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']

models = ['llama_3.1_8b_judge']

for i in range(len(models)):
    model_name = models[i]

    for dataset in datasets:
        judge = LLMJudge(prompt_levels=4)
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

        for eval_passage_file in eval_passages_files:
            output_path = '../attack_results/' + eval_passage_file.split('/')[-1].replace('.jsonl', '') + f'_{model_name}.jsonl'
            
            queries = []
            passages = []
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                for line in input_f:
                    example = json.loads(line)
                    queries.append(queries_dict[example['qids']])
                    passages.append(example['adversarial_passages'])
            
            prompts = judge.generate_prompts_from_pairs(queries, passages)
            scores = judge.generate_scores_from_prompts(prompts)

            assert(len(scores) == len(queries))
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                with open(output_path, 'w', encoding='utf-8') as output_f:
                    index = -1
                    for line in input_f:
                        index += 1
                        example = json.loads(line)
                        example.pop('adversarial_passages')
                        example['score'] = scores[index]
                        output_f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        del judge