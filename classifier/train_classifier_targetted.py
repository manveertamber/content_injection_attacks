from transformers import AutoTokenizer, ModernBertModel, ModernBertConfig
import torch
import transformers
from tqdm import tqdm
import numpy as np
import random

import sys
sys.path.append('..')
from adversarial_passage_generator import AdversarialGenerator

train_adversarial_generator = AdversarialGenerator(
    passages_file=f'../passages/msmarco.tsv', 
    injection_sentences_file=f'../random_sentences/splits/toxigen_train.txt',
)

dev_adversarial_generator = AdversarialGenerator(
    passages_file=f'../passages/msmarco.tsv', 
    injection_sentences_file=f'../random_sentences/splits/toxigen_dev.txt',
)

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
    def __init__(self, queries, rel_samples, adversarial_generator):
        self.queries = queries
        self.rel_samples = rel_samples
        self.adversarial_generator = adversarial_generator

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        org_query = self.queries[index]
        adv_query = gen_random_adv_query(org_query, self.adversarial_generator)

        rel_passage = random.choice(self.rel_samples[index])
        adv_passage = gen_random_adv_passage(rel_passage, self.adversarial_generator)

        return (org_query, adv_query, rel_passage, adv_passage)

class Collator(object):
    def __init__(self, text_maxlength):
        self.tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base', use_fast=True)
        self.text_maxlength = text_maxlength

    def __call__(self, batch):
        org_queries = [example[0] for example in batch]
        adv_queries = [example[1] for example in batch]
        rel_passages = [example[2] for example in batch]
        adv_passages = [example[3] for example in batch]

        p_org_queries = self.tokenizer.batch_encode_plus(
            org_queries,
            max_length=self.text_maxlength,
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

print("BEGIN LOADING TRAINING DATA")

train_queries = []
train_rel_samples = []
with open('../training_data/msmarco_pairs_train.tsv', 'r', encoding='utf-8') as tsv_file:
    lines = tsv_file.readlines()
    random.Random(4).shuffle(lines)
    for line in lines:
        vals = line.split('\t')
        train_queries.append(vals[0].strip())
        rel_samples = []
        for j in range(1, len(vals)):
            rel_samples.append(vals[j].strip())
        train_rel_samples.append(rel_samples)

dev_queries = []
dev_rel_samples = []
with open('../training_data/msmarco_pairs_dev.tsv', 'r', encoding='utf-8') as tsv_file:
    lines = tsv_file.readlines()
    random.Random(4).shuffle(lines)
    for line in lines:
        vals = line.split('\t')
        dev_queries.append(vals[0].strip())
        rel_samples = []
        for j in range(1, len(vals)):
            rel_samples.append(vals[j].strip())
        dev_rel_samples.append(rel_samples)

train_dataset = Dataset(train_queries, train_rel_samples, train_adversarial_generator)
dev_dataset = Dataset(dev_queries, dev_rel_samples, dev_adversarial_generator)

collator_function = Collator(text_maxlength=512)

batch_size = 32  # 32 queries in batch

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size,
    collate_fn=collator_function,
    num_workers=2, 
    shuffle=True,
    drop_last=True,
    pin_memory=True
)

dev_dataloader = torch.utils.data.DataLoader(
    dev_dataset, 
    batch_size=batch_size,
    collate_fn=collator_function,
    num_workers=2, 
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

lr = 1e-5

cross_entropy_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=lr)

def warmup(current_step: int):
    warmup_steps = 50
    if current_step < warmup_steps: 
        return float(current_step / warmup_steps)
    else:
        return 1.0

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

best_loss = float('inf')
best_acc = 0

gradient_accumulation_steps = 2

for epoch in tqdm(range(10)):
    # Evaluate
    classifier_model.eval()
    running_accuracy_sum = 0
    running_loss_sum = 0
    num_dev_samples = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dev_dataloader), leave=False):
            (org_query_ids, org_query_mask,
             adv_query_ids, adv_query_mask,
             rel_passage_ids, rel_passage_mask,
             adv_passage_ids, adv_passage_mask) = batch

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                org_query_classifications = classifier_model(org_query_ids.cuda(), org_query_mask.cuda())
                adv_query_classifications = classifier_model(adv_query_ids.cuda(), adv_query_mask.cuda())
                rel_passage_classifications = classifier_model(rel_passage_ids.cuda(), rel_passage_mask.cuda())
                adv_passage_classifications = classifier_model(adv_passage_ids.cuda(), adv_passage_mask.cuda())

            preds = torch.cat((
                org_query_classifications, 
                adv_query_classifications, 
                rel_passage_classifications, 
                adv_passage_classifications
            ))
            
            current_batch_size = org_query_ids.size(0)
            targets = torch.cat((
                torch.zeros(current_batch_size, dtype=torch.long),
                torch.ones(current_batch_size, dtype=torch.long), 
                torch.zeros(current_batch_size, dtype=torch.long), 
                torch.ones(current_batch_size, dtype=torch.long)
            )).cuda()
            
            running_loss_sum += cross_entropy_loss(preds, targets).detach().cpu() * len(targets)
            predicted_labels = torch.argmax(preds, dim=1)
            running_accuracy_sum += (predicted_labels == targets).sum().detach().cpu()
            num_dev_samples += len(targets)
            
    dev_acc = running_accuracy_sum / num_dev_samples
    dev_loss = running_loss_sum / num_dev_samples
    
    print(f"Epoch {epoch} - DEV ACCURACY: {dev_acc:.4f}")
    print(f"Epoch {epoch} - DEV LOSS: {dev_loss:.4f}")
    print()

    if dev_loss < best_loss:
        best_loss = dev_loss
        torch.save(classifier_model.state_dict(), './best_classifier_model_modernbert_targeted.pth')
    else:
        break

    classifier_model.train()
    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(train_dataloader), leave=False):
        (org_query_ids, org_query_mask,
         adv_query_ids, adv_query_mask,
         rel_passage_ids, rel_passage_mask,
         adv_passage_ids, adv_passage_mask) = batch

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            org_query_classifications = classifier_model(org_query_ids.cuda(), org_query_mask.cuda())
            adv_query_classifications = classifier_model(adv_query_ids.cuda(), adv_query_mask.cuda())
            rel_passage_classifications = classifier_model(rel_passage_ids.cuda(), rel_passage_mask.cuda())
            adv_passage_classifications = classifier_model(adv_passage_ids.cuda(), adv_passage_mask.cuda())

        preds = torch.cat((
            org_query_classifications,
            adv_query_classifications,
            rel_passage_classifications,
            adv_passage_classifications
        ))
        
        targets = torch.cat((
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long),
            torch.zeros(batch_size, dtype=torch.long),
            torch.ones(batch_size, dtype=torch.long)
        )).cuda()
        
        loss = cross_entropy_loss(preds, targets)
        loss = loss / gradient_accumulation_steps

        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

print("Best loss observed during training:", best_loss)
