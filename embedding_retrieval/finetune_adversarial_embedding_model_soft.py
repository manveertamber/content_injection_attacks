import json
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import torch
import transformers
from tqdm import tqdm
import numpy as np
import argparse
import random
from grad_cache.functional import cached, cat_input_tensor

import sys
sys.path.append('..')
from adversarial_passage_generator import AdversarialGenerator

train_adversarial_generator = AdversarialGenerator(
    passages_file=f'../passages/msmarco.tsv', 
    injection_sentences_file=f'../random_sentences/splits/msmarco_train.txt',
)
dev_adversarial_generator = AdversarialGenerator(
    passages_file=f'../passages/msmarco.tsv', 
    injection_sentences_file=f'../random_sentences/splits/msmarco_dev.txt',
)

class Config:
    def __init__(self):
        self.model_name = 'BAAI/bge-base-en-v1.5'
        self.instruction = "Represent this sentence for searching relevant passages: "
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.adj_norm = False

        self.lr = 2e-5
        self.weight_decay = 0.01
        self.batch_size = 512
        self.accumulation_steps = 16
        self.num_epochs = 10
        self.warmup_steps = 10
        self.query_maxlength = 32
        self.text_maxlength = 256
        self.temp = 0.01
        self.temp_softener = 1.1
        
        self.train_file = '../training_data/msmarco_pairs_train.tsv'
        self.dev_file = '../training_data/msmarco_pairs_dev.tsv'
        self.save_model_name = f"new_random_embedding_model_softened_{self.temp_softener}"

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

def gen_random_adv_query(query, adversarial_generator):
    if random.randrange(2):
        # inject query into passage
        if random.randrange(2):
            # take random passage of at least 10 words
            passage = ""
            while len(passage.split()) < 10:
                passage = adversarial_generator.sample_random_passage()
        else:
            # sample 10-100 random words to form a passage
            num_random_words = random.randint(10, 100)
            passage = adversarial_generator.form_text_from_random_words(num_random_words)
        
        location = random.choice(['start', 'middle', 'end'])
        num_injections = random.randint(1, 3)
        return adversarial_generator.inject_query_into_passage(
            passage, query, location, num_injections=num_injections, shorten_passage=False
        )
    
    else:
        if random.randrange(2):
            # take random passage of at least 10 words
            passage = ""
            while len(passage.split()) < 10:
                passage = adversarial_generator.sample_random_passage()
        else:
            # sample 10-100 random words to form a passage
            num_random_words = random.randint(10, 100)
            passage = adversarial_generator.form_text_from_random_words(num_random_words)

        location = random.choice(['start', 'middle', 'end'])
        repeat_count = random.randint(1, 3)
        for repetition_i in range(repeat_count):
            passage = adversarial_generator.inject_query_keywords_into_passage(
                passage, query, location, expand=False, num_synonyms_per_word=1, shorten_passage=False
            )
        return passage

def gen_random_adv_passage(passage, adversarial_generator):
    num_sentences = random.randint(1, 3)
    location = random.choice(['start', 'middle', 'end'])
    return adversarial_generator.inject_random_sentences_into_passage(
        passage, num_sentences, location, shorten_passage=False
    )

class Dataset(torch.utils.data.Dataset):
    def __init__(self, queries, rel_samples, config, adversarial_generator):
        self.queries = queries
        self.rel_samples = rel_samples
        self.config = config
        self.adversarial_generator = adversarial_generator

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        org_query = self.config.instruction + self.queries[index]
        adv_query = gen_random_adv_query(self.queries[index], self.adversarial_generator)

        rel_passage = random.choice(self.rel_samples[index])
        adv_passage = gen_random_adv_passage(rel_passage, self.adversarial_generator)

        return (org_query, adv_query, rel_passage, adv_passage)

class Collator(object):
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.text_maxlength = config.text_maxlength
        self.query_maxlength = config.query_maxlength

    def __call__(self, batch):
        org_queries = [example[0] for example in batch]
        adv_queries = [example[1] for example in batch]
        rel_passages = [example[2] for example in batch]
        adv_passages = [example[3] for example in batch]

        p_org_queries = self.tokenizer.batch_encode_plus(
            org_queries,
            max_length=self.query_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )
        p_adv_queries = self.tokenizer.batch_encode_plus(
            adv_queries,
            max_length=self.text_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )
        p_rel_passages = self.tokenizer.batch_encode_plus(
            rel_passages,
            max_length=self.text_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )
        p_adv_passages = self.tokenizer.batch_encode_plus(
            adv_passages,
            max_length=self.text_maxlength,
            padding=True, 
            return_tensors='pt',
            truncation=True
        )

        return (p_org_queries['input_ids'], p_org_queries['attention_mask'],
                p_adv_queries['input_ids'], p_adv_queries['attention_mask'],
                p_rel_passages['input_ids'], p_rel_passages['attention_mask'],
                p_adv_passages['input_ids'], p_adv_passages['attention_mask'])

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
        
    train_dataset = Dataset(train_queries, train_rel_samples, config, train_adversarial_generator)

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
        
    dev_dataset = Dataset(dev_queries, dev_rel_samples, config, dev_adversarial_generator)
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
    
    @cat_input_tensor
    def adversarial_retrieval_loss(query_embeddings, adv_query_embeddings, adv_passage_embeddings, rel_passage_embeddings):        
        temp = torch.tensor(config.temp).double().cuda()
        
        adv_query_scores =  torch.matmul(adv_query_embeddings, query_embeddings.T).double()
        adv_query_scores = torch.exp(adv_query_scores / (temp*config.temp_softener))
        
        adv_passage_scores = torch.matmul(adv_passage_embeddings, query_embeddings.T).double()
        adv_passage_scores = torch.exp(adv_passage_scores / (temp*config.temp_softener))
        
        rel_passage_scores = torch.matmul(rel_passage_embeddings, query_embeddings.T).double()
        rel_passage_scores = torch.exp(rel_passage_scores / temp)
        
        adv_rel_passage_scores = torch.matmul(adv_passage_embeddings, rel_passage_embeddings.T).double()
        adv_rel_passage_scores = torch.exp(adv_rel_passage_scores / (temp*config.temp_softener)) 

        losses = torch.diagonal(rel_passage_scores) / (torch.sum(rel_passage_scores, dim=0) + torch.sum(adv_passage_scores, dim=0) + torch.sum(adv_query_scores, dim=0)
                                                + torch.sum(adv_rel_passage_scores, dim=0))
        losses = torch.log(losses)
        return -losses.mean().float()

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
        cache_adv_query_embeddings = []
        cache_rel_passage_embeddings = []
        cache_adv_passage_embeddings = []

        running_adv_loss_sum = 0
        running_retrieval_loss_sum = 0

        num_dev_batches = 0
        with torch.no_grad():        
            for step, batch in tqdm(enumerate(dev_dataloader)):
                (org_query_ids, org_query_mask,
                adv_query_ids, adv_query_mask,
                rel_passage_ids, rel_passage_mask,
                adv_passage_ids, adv_passage_mask) = batch

                query_embeddings = emb_model(org_query_ids.cuda(), org_query_mask.cuda(), query=True)
                adv_query_embeddings = emb_model(adv_query_ids.cuda(), adv_query_mask.cuda())
                rel_passage_embeddings = emb_model(rel_passage_ids.cuda(), rel_passage_mask.cuda())
                adv_passage_embeddings = emb_model(adv_passage_ids.cuda(), adv_passage_mask.cuda())
                
                cache_query_embeddings.append(query_embeddings)
                cache_adv_query_embeddings.append(adv_query_embeddings)
                cache_rel_passage_embeddings.append(rel_passage_embeddings)
                cache_adv_passage_embeddings.append(adv_passage_embeddings)

                if (step + 1) % accumulation_steps == 0 or (step < accumulation_steps and (step + 1) == len(dev_dataloader)):
                    num_dev_batches += 1

                    retrieval_loss = contrastive_retrieval_loss(cache_query_embeddings, cache_rel_passage_embeddings).detach().cpu()
                    adv_loss = adversarial_retrieval_loss(cache_query_embeddings, cache_adv_query_embeddings, cache_adv_passage_embeddings, cache_rel_passage_embeddings).detach().cpu()
                    
                    cache_query_embeddings = []
                    cache_adv_query_embeddings = []
                    cache_rel_passage_embeddings = []
                    cache_adv_passage_embeddings = []
                    
                    running_retrieval_loss_sum += retrieval_loss
                    running_adv_loss_sum += adv_loss
                    
        print("RETRIEVAL LOSS", running_retrieval_loss_sum / num_dev_batches)
        print("ADVERSARIAL LOSS", running_adv_loss_sum / num_dev_batches)
        dev_loss = running_adv_loss_sum / num_dev_batches

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            emb_model.bert.save_pretrained(config.save_model_name)
            print(config.save_model_name)
        else:
            quit()

        emb_model.train()
        
        cache_query_embeddings = []
        cache_adv_query_embeddings = []
        cache_rel_passage_embeddings = []
        cache_adv_passage_embeddings = []

        closures_q = []
        closures_aq = []
        closures_rp = []
        closures_ap = []

        num_train_batches = 0

        for step, batch in tqdm(enumerate(train_dataloader)):
            (org_query_ids, org_query_mask,
            adv_query_ids, adv_query_mask,
            rel_passage_ids, rel_passage_mask,
            adv_passage_ids, adv_passage_mask) = batch
            
            query_embeddings, cq = call_model(emb_model, org_query_ids.cuda(), org_query_mask.cuda(), query=True)
            adv_query_embeddings, caq = call_model(emb_model, adv_query_ids.cuda(), adv_query_mask.cuda())
            rel_passage_embeddings, crp = call_model(emb_model, rel_passage_ids.cuda(), rel_passage_mask.cuda())
            adv_passage_embeddings, cap = call_model(emb_model, adv_passage_ids.cuda(), adv_passage_mask.cuda())

            cache_query_embeddings.append(query_embeddings)
            cache_adv_query_embeddings.append(adv_query_embeddings)
            cache_rel_passage_embeddings.append(rel_passage_embeddings)
            cache_adv_passage_embeddings.append(adv_passage_embeddings)

            closures_q.append(cq)
            closures_aq.append(caq)
            closures_rp.append(crp)
            closures_ap.append(cap)
            
            if (step + 1) % accumulation_steps == 0:
                num_train_batches += 1

                loss = adversarial_retrieval_loss(cache_query_embeddings, cache_adv_query_embeddings, cache_adv_passage_embeddings, cache_rel_passage_embeddings)
                loss.backward()

                for f, r in zip(closures_q, cache_query_embeddings):
                    f(r)
                for f, r in zip(closures_aq, cache_adv_query_embeddings):
                    f(r)
                for f, r in zip(closures_rp, cache_rel_passage_embeddings):
                    f(r)
                for f, r in zip(closures_ap, cache_adv_passage_embeddings):
                    f(r)

                cache_query_embeddings = []
                cache_adv_query_embeddings = []
                cache_rel_passage_embeddings = []
                cache_adv_passage_embeddings = []

                closures_q = []
                closures_aq = []
                closures_rp = []
                closures_ap = []

                torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

if __name__ == "__main__":
    main()