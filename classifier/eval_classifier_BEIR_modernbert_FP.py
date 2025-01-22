from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig
import torch
import transformers
from tqdm import tqdm
import numpy as np
import random
from datasets import load_dataset


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

beir_datasets = ['msmarco', 'fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'dbpedia-entity',  'climate-fever']

for dataset in beir_datasets:
    print("dataset", dataset)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, passages):
            self.passages = passages

        def __len__(self):
            return len(self.passages)

        def __getitem__(self, index):
            example = self.passages[index]
            docid = example['docid']
            text = example['text']

            return {
                'docid': str(docid),
                'text': text,
            }

    class Collator(object):
        def __init__(self, tokenizer, text_maxlength):
            self.tokenizer = tokenizer
            self.text_maxlength = text_maxlength

        def __call__(self, batch):
            texts = [example['text'] for example in batch]
            docids = [example['docid'] for example in batch]

            p = self.tokenizer.batch_encode_plus(
                texts,
                max_length=self.text_maxlength,
                padding=True,
                return_tensors='pt',
                truncation=True
            )

            return (p['input_ids'], p['attention_mask'].bool(), docids)

    passages = []
    ds = load_dataset("BeIR/" + dataset, "corpus")['corpus']; id_key='_id'
    for line in tqdm(ds):        
        passage_dict = {}
        passage_dict['docid'] = str(line[id_key])
        passage_dict['text'] = str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        passages.append(passage_dict)

    passage_dataset = Dataset(passages)
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    collator_function = Collator(tokenizer, text_maxlength=512)

    batch_size=512

    dataloader = torch.utils.data.DataLoader(
        passage_dataset,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collator_function,
        shuffle=False,
        drop_last=False,
    )

    num_error = 0
    num_samples = 0
    error_ids = [] 
    for i, batch in tqdm(enumerate(dataloader)):
        (passage_ids, passage_masks, docids) = batch

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                preds = classifier_model(passage_ids.cuda(), passage_masks.cuda()).float().detach().cpu()

        predicted_labels = torch.argmax(preds, dim=1)
        misclassified = (predicted_labels == 1).nonzero(as_tuple=True)[0]
        error_ids.extend([docids[idx] for idx in misclassified])
        num_error += len(misclassified)
        num_samples += len(docids)

    print("NUM_ERROR", num_error)
    print("num_samples", num_samples)

    with open(f'untargeted_modernbert_errors_{dataset}.txt', 'w') as f:
        for docid in error_ids:
            f.write(docid + '\n')