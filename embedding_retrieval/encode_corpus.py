# python3 encode_beir_corpus_norm.py --model_name models/adversarial-targeted-1.0-retriever --normalize --pooling cls --batch_size 1800

import json
from transformers import AutoModel, AutoTokenizer
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
import argparse
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset

class EmbeddingModelWrapper(torch.nn.Module):
    def __init__(self, model_name, normalize=True, pooling="cls"):
        super(EmbeddingModelWrapper, self).__init__()
        if pooling == "LLM":
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.normalize = normalize
        self.pooling = pooling

    def forward(self, ids, mask):
        model_output = self.model(ids, mask)
        if self.pooling == "mean":
            embs = self.average_pool(model_output.last_hidden_state, mask)  # Mean pooling
        elif self.pooling == "LLM":
            embs = self.last_token_pool(model_output.last_hidden_state, mask)
        else:
            assert(self.pooling == "cls")
            embs = model_output[0][:, 0]  # CLS token representation
        
        if self.normalize:
            norms = torch.norm(embs, dim=1, keepdim=True)
            embs = embs / (norms)

        return embs

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the embedding model to use')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the embeddings')
    parser.add_argument('--pooling', type=str, choices=['cls', 'mean', 'LLM'], default='cls', help='Pooling method to use for embeddings')
    parser.add_argument('--batch_size', type=int, default=1800, help='Batch Size')
    parser.add_argument('--dim', type=int, default=768, help='Embedding Dimension')
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    emb_model = EmbeddingModelWrapper(args.model_name, normalize=args.normalize, pooling=args.pooling).cuda().eval()


    datasets = ["msmarco", "trec-covid", "nfcorpus",  "fiqa", "dbpedia-entity", "climate-fever", "scifact"]

    for dataset_i in range(len(datasets)):
        dataset = datasets[dataset_i]

        with open('passage_data/' + dataset + '_passage_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        passage_dataset = Dataset(data)
        collator_function = Collator(tokenizer, text_maxlength=512)

        dataloader = torch.utils.data.DataLoader(
            passage_dataset,
            batch_size=args.batch_size,
            num_workers=8,
            collate_fn=collator_function,
            shuffle=False,
            drop_last=False,
        )

        index = faiss.IndexFlatIP(768)
        model_path = args.model_name.replace('/', '_')

        if not os.path.exists('attack_indices/' + model_path + "_" + dataset + '_index'):
            os.mkdir('attack_indices/' + model_path + "_" + dataset + '_index')

        with torch.no_grad():
            docid_file = open('attack_indices/' + model_path + "_" + dataset + '_index/docid', 'w')

            for i, batch in tqdm(enumerate(dataloader)):
                (passage_ids, passage_masks, docids) = batch

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):    
                    embeddings = emb_model(passage_ids.cuda(), passage_masks.cuda()).float().detach().cpu().numpy()

                for emb in embeddings:
                    index.add(emb.astype(np.float32).reshape((1,768)))

                for docid in docids:
                    docid_file.write(docid + '\n')


            faiss.write_index(index, 'attack_indices/' + model_path + "_" + dataset + '_index/index')
            docid_file.close()

if __name__ == "__main__":
    main()
