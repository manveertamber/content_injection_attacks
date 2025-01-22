import sys
sys.path.append('..')
from openai import AzureOpenAI
from tqdm import tqdm
import json
from llm_judge import LLMJudge


client = AzureOpenAI(
  azure_endpoint = '', 
  api_key='',  
  api_version="2024-08-01-preview"
)

datasets = ['dl19', 'dl20']
models = ['gpt4o', 'gpt4o-adv']

for model_name in models:
    if 'adv' in model_name:
        judge = LLMJudge(model_name=model_name, adversarial=True)
    else:
        judge = LLMJudge(model_name=model_name)

    for dataset in datasets:
        print("dataset", dataset)
        eval_passages_files = [f'../../attack_passages/passages/keyword_injection_general_passages_{dataset}.jsonl',
                            f'../../attack_passages/passages/query_injection_general_passages_{dataset}.jsonl',
                            f'../../attack_passages/passages/generated_targeted_sentence_injection_{dataset}.jsonl',
                            f'../../attack_passages/passages/generated_random_sentence_injection_{dataset}.jsonl',
                            f'../../attack_passages/passages/keyword_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                            f'../../attack_passages/passages/query_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                            f'../../attack_passages/passages/random_sentence_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                            f'../../attack_passages/passages/targeted_sentence_injection_SEO_passages_{model_name}_{dataset}.jsonl',
                                ]

        queries_dict = {}
        with open(f'../../queries/{dataset}-queries.tsv', 'r', encoding='utf-8') as f:
            for line in f:
                vals = line.split('\t')
                queries_dict[vals[0].strip()] =  vals[1].strip()

        for eval_passage_file in eval_passages_files:
            output_path = '../../attack_results/' + eval_passage_file.split('/')[-1].replace('.jsonl', '') + f'_{model_name}.jsonl'

            queries = []
            passages = []
            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                for line in input_f:
                    example = json.loads(line)
                    queries.append(queries_dict[example['qids']])
                    passages.append(example['adversarial_passages'])
            
            prompts = judge.generate_prompts_from_pairs(queries, passages)

            with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
                with open(output_path, 'w', encoding='utf-8') as output_f:
                    index = -1
                    for line in tqdm(input_f):
                        index += 1
                        example = json.loads(line)
                        qid = example['qids']
                        example.pop('adversarial_passages')
                        
                        prompt = prompts[index]

                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "user", "content": prompt}
                                ]
                            )
                        
                            score = int(response.choices[0].message.content)
                        except Exception as e: 
                            print("FAILED")
                            print(e)
                            score = 0

                        example['score'] = score
                        output_f.write(json.dumps(example, ensure_ascii=False) + '\n')

