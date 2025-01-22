import os
import csv
import random
import spacy
from datasets import load_dataset
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["ner"])

def is_valid_sentence(sent) -> bool:
    text = sent.text.strip()

    # (1) Check if sentence is too short
    if len(text) <= 30:
        return False
    
    # (2) Check if sentence is too long (300 characters)
    if len(text) > 300:
        return False

    # (3) Check token count
    if len(sent) <= 5:
        return False

    # (4) Must contain at least one verb
    if not any(token.pos_ == "VERB" for token in sent):
        return False

    # (5) Remove stopwords and punctuation to see how many tokens remain
    non_stop_tokens = [t for t in sent if not t.is_stop and t.is_alpha]
    if len(non_stop_tokens) < 3:
        return False

    # (6) Must contain at least one NOUN or PROPN
    if not any(token.pos_ in ["NOUN", "PROPN"] for token in sent):
        return False

    return True
    
def extract_sentences_from_texts(texts, batch_size=128, n_process=1):
    all_sentences = []
    
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    for doc in docs:
        valid_sents = [sent.text.strip() for sent in doc.sents if is_valid_sentence(sent)]
        all_sentences.extend(valid_sents)
    
    return all_sentences

def generate_toxigen_sentences():
    output_dir = "toxigen_output"
    os.makedirs(output_dir, exist_ok=True)

    generations_train = load_dataset("toxigen/toxigen-data", "train", split="train")
    filtered_generations = generations_train.filter(lambda x: x["prompt_label"] == 1)
    generation_texts = [row["generation"].replace('\n', ' ').replace('\r', ' ').strip() for row in filtered_generations]
    
    print("Extracting valid sentences from 'generations' train split...")
    generation_sentences = extract_sentences_from_texts(
        generation_texts,
        batch_size=128,
        n_process=8
    )

    random.shuffle(generation_sentences)

    train_file = os.path.join(output_dir, "train.txt")
    print(f"Writing {len(generation_sentences)} sentences to {train_file}")
    with open(train_file, "w", encoding="utf-8") as f:
        for sent in generation_sentences:
            f.write(sent + "\n")

    annotated_test = load_dataset("toxigen/toxigen-data", "annotated", split="test")
    filtered_annotated = annotated_test.filter(
        lambda x: x["toxicity_ai"] >= 4 or x["toxicity_human"] >= 4
    )
    annotated_texts = [row["text"].replace('\n', ' ').replace('\r', ' ').strip() for row in filtered_annotated]

    print("Extracting valid sentences from 'annotated' test split...")
    annotated_sentences = extract_sentences_from_texts(
        annotated_texts,
        batch_size=128,
        n_process=8
    )
    test_file = os.path.join(output_dir, "test.txt")
    print(f"Writing {len(annotated_sentences)} sentences to {test_file}")
    with open(test_file, "w", encoding="utf-8") as f:
        for sent in annotated_sentences:
            f.write(sent + "\n")

    print("Done!")

if __name__ == "__main__":
    generate_toxigen_sentences()
