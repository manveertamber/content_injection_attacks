import sys
sys.path.append('..')

from datasets import load_dataset
import json
from adversarial_passage_generator import AdversarialGenerator
import random
from tqdm import tqdm

datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']
for dataset in tqdm(datasets):

    print('dataset', dataset)
    output_filepath = f'passages/query_injection_general_passages_{dataset}.jsonl'
    print('output_filepath', output_filepath)
    
    adversarial_generator = AdversarialGenerator(
        passages_file=f'../passages/msmarco.tsv',
        injection_sentences_file='../random_sentences/splits/msmarco_test.txt', 
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
        "num_query_repetitions": [],
        "num_original_words": [],
        "locations": [],
        "passage_types": [],
    }

    for qid, query in tqdm(queries_dict.items()):
        query_len = len(query.split())

        sample_count=0

        for trial in range(5):
            random_passage = adversarial_generator.sample_random_passage()
            num_random_passage_words = len(random_passage.split())
            random_words_passage = adversarial_generator.form_text_from_random_words(num_random_passage_words)

            attack_id = f"{qid}_randpassage_trial_{trial}_control"
            outputs['attack_ids'].append(attack_id)
            outputs['qids'].append(qid)
            outputs['adversarial_passages'].append(random_passage)
            outputs["num_injected_words"].append(0)
            outputs['num_query_repetitions'].append(0)
            outputs["num_original_words"].append(num_random_passage_words)    
            outputs["locations"].append('none')
            outputs["passage_types"].append('randpassage')

            attack_id = f"{qid}_randwords_trial_{trial}_control"
            outputs['attack_ids'].append(attack_id)
            outputs['qids'].append(qid)
            outputs['adversarial_passages'].append(random_words_passage)
            outputs["num_injected_words"].append(0)
            outputs['num_query_repetitions'].append(0)
            outputs["num_original_words"].append(num_random_passage_words)    
            outputs["locations"].append('none')
            outputs["passage_types"].append('randwords')

            for query_repetition in [1, 2, 3]:
                for location in ['start', 'middle', 'end']:
                    attack_id = f"{qid}_randpassage_trial_{trial}_sample_{sample_count}"
                    attack_passage = adversarial_generator.inject_query_into_passage(random_passage, query, location, num_injections=query_repetition)
                    outputs['attack_ids'].append(attack_id)
                    outputs['qids'].append(qid)
                    outputs['adversarial_passages'].append(attack_passage)
                    outputs["num_injected_words"].append(query_len * query_repetition)
                    outputs['num_query_repetitions'].append(query_repetition)
                    outputs["num_original_words"].append(num_random_passage_words)    
                    outputs["locations"].append(location)
                    outputs["passage_types"].append('randpassage')

                    attack_id = f"{qid}_randwords_trial_{trial}_sample_{sample_count}"
                    attack_passage = adversarial_generator.inject_query_into_passage(random_words_passage, query, location, num_injections=query_repetition)
                    outputs['attack_ids'].append(attack_id)
                    outputs['qids'].append(qid)
                    outputs['adversarial_passages'].append(attack_passage)
                    outputs["num_injected_words"].append(query_len * query_repetition)
                    outputs['num_query_repetitions'].append(query_repetition)
                    outputs["num_original_words"].append(num_random_passage_words)    
                    outputs["locations"].append(location)
                    outputs["passage_types"].append('randwords')

                    sample_count+=1


    with open(output_filepath, 'w') as log_file:
        for i in range(len(outputs["attack_ids"])):
            entry = {key: outputs[key][i] for key in outputs}
            log_file.write(json.dumps(entry) + '\n')

    print('num_queries', len(queries_dict.keys()))
   