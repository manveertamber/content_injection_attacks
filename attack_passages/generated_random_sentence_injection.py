import sys
sys.path.append('..')

from datasets import load_dataset
import json
from adversarial_passage_generator import AdversarialGenerator
import random
from tqdm import tqdm

datasets = ['dl19', 'dl20']
lengths = [50, 100, 200]

for dataset in tqdm(datasets):
    generated_passages_dict = {}
    for length in lengths:
        generated_passages_dict[length] = {}
        filepath = f'../rel_passage_gen/generated_passages/{dataset}_generated_passages_{length}_words.tsv'
        with open(filepath, 'r', encoding='utf-8') as generated_passages_file:
            for line in generated_passages_file:
                try:
                    qid, generated_passage = line.strip().split('\t')
                    generated_passages_dict[length][qid] = generated_passage.strip()
                except:
                    print("Problem parsing line:", line)
                    sys.exit(1)

    print('dataset', dataset)
    output_filepath = f'passages/generated_random_sentence_injection_{dataset}.jsonl'
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
        "num_sentence_injection": [],
        "num_original_words": [],
        "locations": [],
        "passage_types": [],
        "passage_repetition": [],
        "passage_length": []
    }

    for qid, query in tqdm(queries_dict.items()):
        query_len = len(query.split())

        for length in lengths:
            if qid not in generated_passages_dict[length]:
                continue

            generated_positive_passage = generated_passages_dict[length][qid]
            generated_positive_passage_num_words = len(generated_positive_passage.split())

            # 1) Control "no injection" passage
            attack_id = f"{qid}_gen_pos_passage_control_{length}"
            outputs['attack_ids'].append(attack_id)
            outputs['qids'].append(qid)
            outputs['adversarial_passages'].append(generated_positive_passage)
            outputs["num_injected_words"].append(0)
            outputs['num_sentence_injection'].append(0)
            outputs["num_original_words"].append(generated_positive_passage_num_words)
            outputs["locations"].append('none')
            outputs["passage_types"].append('gen_pos_passage')
            outputs['passage_repetition'].append(1)
            outputs['passage_length'].append(length)

            sample_count = 0
            # 2) Generate adversarial passages
            for trial in range(5):
                for num_sentence_injection in [1, 2]:
                    inject_sentences = adversarial_generator.get_random_sentences(num_sentence_injection)

                    for location in ['start', 'middle', 'end']:
                        if location == 'start' and length < 200:
                            passage_repetition = 2
                            attack_id = f"{qid}_gen_pos_passage_trial_{trial}_sample_{sample_count}_{length}"
                            attack_passage = adversarial_generator.inject_sentences(
                                generated_positive_passage + " " + generated_positive_passage, 
                                inject_sentences, 
                                location
                            )
                            outputs['attack_ids'].append(attack_id)
                            outputs['qids'].append(qid)
                            outputs['adversarial_passages'].append(attack_passage)
                            outputs["num_injected_words"].append(
                                len(attack_passage.split()) - 2 * generated_positive_passage_num_words
                            )
                            outputs['num_sentence_injection'].append(num_sentence_injection)
                            outputs["num_original_words"].append(2 * generated_positive_passage_num_words)
                            outputs["locations"].append(location)
                            outputs["passage_types"].append('gen_pos_passage')
                            outputs['passage_repetition'].append(passage_repetition)
                            outputs['passage_length'].append(length)

                        passage_repetition = 1
                        attack_id = f"{qid}_gen_pos_passage_trial_{trial}_sample_{sample_count}_{length}"
                        attack_passage = adversarial_generator.inject_sentences(
                            generated_positive_passage, inject_sentences, location
                        )
                        outputs['attack_ids'].append(attack_id)
                        outputs['qids'].append(qid)
                        outputs['adversarial_passages'].append(attack_passage)
                        outputs["num_injected_words"].append(
                            len(attack_passage.split()) - generated_positive_passage_num_words
                        )
                        outputs['num_sentence_injection'].append(num_sentence_injection)
                        outputs["num_original_words"].append(generated_positive_passage_num_words)
                        outputs["locations"].append(location)
                        outputs["passage_types"].append('gen_pos_passage')
                        outputs['passage_repetition'].append(passage_repetition)
                        outputs['passage_length'].append(length)

                        sample_count += 1

    with open(output_filepath, 'w', encoding='utf-8') as log_file:
        for i in range(len(outputs["attack_ids"])):
            entry = {key: outputs[key][i] for key in outputs}
            log_file.write(json.dumps(entry) + '\n')

    print('num_queries:', len(queries_dict))