import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import torch
import transformers
from tqdm import tqdm
import numpy as np
import pickle
import argparse
import random
import time
from grad_cache.functional import cached, cat_input_tensor

class Config:
    def __init__(self):
        self.model_name = 'BAAI/bge-base-en-v1.5'
        self.save_model_name = "plain_embedding_model_adj_norm"
        self.instruction = "Represent this sentence for searching relevant passages: "
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1

        self.lr = 2e-5
        self.weight_decay = 0.01
        self.batch_size = 1024
        self.accumulation_steps = 8
        self.num_epochs = 10
        self.warmup_steps = 10
        self.query_maxlength = 32
        self.text_maxlength = 256
        self.temp = 0.01

        self.train_file = '../training_data/training_pairs_train.tsv'
        self.dev_file = '../training_data/training_pairs_dev.tsv'

        self.adj_norm = False

class Model(torch.nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        self.config = config

        configuration = AutoConfig.from_pretrained(config.model_name)

        configuration.hidden_dropout_prob = config.hidden_dropout_prob
        configuration.attention_probs_dropout_prob = config.attention_probs_dropout_prob

        print("HIDDEN DROPOUT PROB", configuration.hidden_dropout_prob)
        print("ATTENTION PROBS DROPOUT PROB", configuration.attention_probs_dropout_prob)

        self.bert = AutoModel.from_pretrained(config.model_name, config=configuration)
        self.bert.gradient_checkpointing_enable()

    def forward(self, ids, mask, query=False):
        embs = self.bert(ids, mask)[0][:, 0]
        
        if query:
            return torch.nn.functional.normalize(embs, p=2, dim=1)
        
        if self.config.adj_norm:
            norms = torch.norm(embs, dim=1, keepdim=True)
            return embs / (1 + norms)
        else:
            return torch.nn.functional.normalize(embs, p=2, dim=1)

@torch.amp.autocast('cuda', dtype=torch.bfloat16)
@cached
def call_model(model, ids, mask, query=False):
    return model(ids, mask, query)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, queries, rel_samples, config):
        self.queries = queries
        self.rel_samples = rel_samples
        self.config = config

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.config.instruction + self.queries[index]
        return (query, random.choice(self.rel_samples[index]))

class Collator(object):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.text_maxlength = config.text_maxlength
        self.query_maxlength = config.query_maxlength

    def __call__(self, batch):
        query_texts = [example[0] for example in batch]
        rel_texts = [example[1] for example in batch]

        p_queries = self.tokenizer.batch_encode_plus(
            query_texts,
            max_length=self.query_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )

        p_rel = self.tokenizer.batch_encode_plus(
            rel_texts,
            max_length=self.text_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )
        
        return (p_queries['input_ids'], p_queries['attention_mask'], p_rel['input_ids'], p_rel['attention_mask'])


def main():
    config = Config()

    emb_model = Model(config)
    emb_model.cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    print("BEGIN LOADING TRAINING DATA")

    train_queries = []
    train_rel_samples = []
    with open(config.train_file, 'r', encoding='utf-8') as tsv_file:
        lines = tsv_file.readlines()
        random.shuffle(lines)
        for line in (lines):
            vals = line.split('\t')
            train_queries.append(vals[0].strip())
            rel_samples = []
            for j in range(1, len(vals)):
                rel_samples.append(vals[j].strip())
            train_rel_samples.append(rel_samples)
        
    train_dataset = Dataset(train_queries, train_rel_samples, config)

    dev_queries = []
    dev_rel_samples = []
    with open(config.dev_file, 'r', encoding='utf-8') as tsv_file:
        lines = tsv_file.readlines()
        random.shuffle(lines)
        for line in (lines):
            vals = line.split('\t')
            dev_queries.append(vals[0].strip())
            rel_samples = []
            for j in range(1, len(vals)):
                rel_samples.append(vals[j].strip())
            dev_rel_samples.append(rel_samples)
        
    dev_dataset = Dataset(dev_queries, dev_rel_samples, config)
    collator_function = Collator(tokenizer, config)

    batch_size = config.batch_size
    accumulation_steps = config.accumulation_steps

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        collate_fn=collator_function,
        num_workers=4, 
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=batch_size,
        collate_fn=collator_function,
        num_workers=4, 
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )
    @cat_input_tensor
    def contrastive_retrieval_loss(query_embeddings, rel_passage_embeddings):        
        temp = torch.tensor(config.temp).double().cuda()

        rel_scores = torch.matmul(rel_passage_embeddings, query_embeddings.T).double()
        rel_scores = torch.exp(rel_scores / temp)

        losses = torch.log(torch.diagonal(rel_scores) / (torch.sum(rel_scores, dim=0))).float()
        return -losses.mean()
    
    optimizer = torch.optim.AdamW(emb_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def warmup(current_step: int):
        if current_step < config.warmup_steps: 
            return float(current_step / config.warmup_steps)
        else:
            return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

    best_dev_loss = float('inf')
    for epoch in range(config.num_epochs):
        emb_model.eval()

        cache_query_embeddings = []
        cache_rel_passage_embeddings = []

        running_loss_sum = 0
        num_dev_batches = 0
        with torch.no_grad():        
            for step, batch in tqdm(enumerate(dev_dataloader)):
                (query_ids, query_mask, rel_ids, rel_mask) = batch

                query_embeddings = emb_model(query_ids.cuda(), query_mask.cuda(), query=True)
                rel_passage_embeddings = emb_model(rel_ids.cuda(), rel_mask.cuda())
                
                cache_query_embeddings.append(query_embeddings)
                cache_rel_passage_embeddings.append(rel_passage_embeddings)

                if (step + 1) % accumulation_steps == 0 or (step < accumulation_steps and (step + 1) == len(dev_dataloader)):
                    num_dev_batches += 1

                    loss = contrastive_retrieval_loss(cache_query_embeddings, cache_rel_passage_embeddings).detach().cpu()
        
                    running_loss_sum += loss

        dev_loss = running_loss_sum / num_dev_batches
        print("RETRIEVAL LOSS", dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            emb_model.bert.save_pretrained(config.save_model_name)
            print(config.save_model_name)
        else:
            quit()

        emb_model.train()
        
        cache_query_embeddings = []
        cache_rel_passage_embeddings = []
        closures_query = []
        closures_rel = []

        num_train_batches = 0

        for step, batch in tqdm(enumerate(train_dataloader)):
            (query_ids, query_mask, rel_ids, rel_mask) = batch
            
            query_embeddings, cq = call_model(emb_model, query_ids.cuda(), query_mask.cuda(), query=True)
            rel_passage_embeddings, cr = call_model(emb_model, rel_ids.cuda(), rel_mask.cuda())

            cache_query_embeddings.append(query_embeddings)
            cache_rel_passage_embeddings.append(rel_passage_embeddings)

            closures_query.append(cq)
            closures_rel.append(cr)

            if (step + 1) % accumulation_steps == 0:
                num_train_batches += 1

                loss = contrastive_retrieval_loss(cache_query_embeddings, cache_rel_passage_embeddings)
                loss.backward()

                for f, r in zip(closures_query, cache_query_embeddings):
                    f(r)
                for f, r in zip(closures_rel, cache_rel_passage_embeddings):
                    f(r)

                cache_query_embeddings = []
                cache_rel_passage_embeddings = []
                closures_query = []
                closures_rel = []

                torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

if __name__ == "__main__":
    main()