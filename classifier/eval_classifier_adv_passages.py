from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig
import torch
import transformers
from tqdm import tqdm
import numpy as np
import random
from datasets import load_dataset
import json

datasets = ['dl19', 'dl20', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity', 'climate-fever']
batch_size = 256

tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        configuration = ModernBertConfig.from_pretrained('answerdotai/ModernBERT-base')
        configuration.attention_dropout = 0.1
        
        self.bert = ModernBertModel.from_pretrained('answerdotai/ModernBERT-base', config=configuration)

        self.linear1 = torch.nn.Linear(768, 2)

    def forward(self, ids, mask):
        unnormalized_cls = self.bert(ids, mask)[0][:, 0]
        output = self.linear1(unnormalized_cls)
        return output

classifier_model = Model().cuda()
classifier_model.load_state_dict(torch.load('./best_classifier_model_modernbert.pth'))
classifier_model.eval()


model_name = 'untargeted_classifier'
for dataset in datasets:
    print("dataset", dataset)
    eval_passages_files = [f'../attack_passages/passages/keyword_injection_general_passages_{dataset}.jsonl',
                            f'../attack_passages/passages/query_injection_general_passages_{dataset}.jsonl',
                            ]

    eval_passages_files += [f'../attack_passages/passages/random_sentence_injection_SEO_passages_bge-base-retriever_{dataset}.jsonl',
                            f'../attack_passages/passages/targeted_sentence_injection_SEO_passages_bge-base-retriever_{dataset}.jsonl']
    
    for eval_passage_file in eval_passages_files:
        output_path = '../attack_results/' + eval_passage_file.split('/')[-1].replace('.jsonl', '') + f'_{model_name}.jsonl'
        
        passages = []

        with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
            for line in input_f:
                example = json.loads(line)
                if 'current_ranks_scores' in example:
                    if example['current_ranks_scores'][0] == 1:
                        passages.append(example['adversarial_passages'])
                else:
                    passages.append(example['adversarial_passages'])

        passage_classifications = []
        for i in tqdm(range(0, len(passages), batch_size)):
            with torch.no_grad():
                
                p = tokenizer.batch_encode_plus(
                    passages[i:i+batch_size],
                    max_length=512,
                    padding=True,
                    return_tensors='pt',
                    truncation=True
                )
                preds = classifier_model(p['input_ids'].cuda(), p['attention_mask'].cuda()).cpu().detach()
                predicted_labels = torch.argmax(preds, dim=1).tolist()
                passage_classifications.extend(predicted_labels)

        assert(len(passage_classifications) == len(passages))
        with open(eval_passage_file, 'r', encoding='utf-8') as input_f:
            with open(output_path, 'w', encoding='utf-8') as output_f:
                index = -1
                for line in input_f:
                    example = json.loads(line)
                    if 'current_ranks_scores' in example:
                        if example['current_ranks_scores'][0] != 1:
                            continue
                    index += 1
                    qid = example['qids']
                    example.pop('adversarial_passages')
                    example['classification'] = passage_classifications[index]
                    output_f.write(json.dumps(example, ensure_ascii=False) + '\n')
