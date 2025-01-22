import os
import random
import spacy
from datasets import load_dataset
from tqdm import tqdm
import csv

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


def generate_files_from_beir(corpus_name: str):
    """
    Loads a BEIR corpus, filters passages for valid sentences (based on is_valid_sentence criteria),
    and writes out TSV and TXT files (without generating smaller subsets).
    """
    passages_dir = "passages/"
    sentences_dir = "random_sentences/"
    os.makedirs(passages_dir, exist_ok=True)
    os.makedirs(sentences_dir, exist_ok=True)

    passages_file = os.path.join(passages_dir, f"{corpus_name}.tsv")
    sentences_file = os.path.join(sentences_dir, f"{corpus_name}.txt")

    ds = load_dataset(f"BeIR/{corpus_name}", "corpus")["corpus"]

    sentences = []
    filtered_passages = []

    with open(passages_file, "w", encoding="utf-8", newline='') as pf:
        tsv_writer = csv.writer(pf, delimiter='\t')
        tsv_writer.writerow(["id", "passage"])  # Header row

        passage_texts = []
        passage_ids = []

        # Collect passage texts for batch processing
        for line in tqdm(ds, desc="Reading corpus"):
            passage_id = str(line["_id"])
            # Truncate passage text to avoid extremely long docs
            passage_text = (
                str(line["text"])
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
                .strip()[:2048]
            )
            passage_ids.append(passage_id)
            passage_texts.append(passage_text)

        # Process passages in batches with spaCy
        docs = nlp.pipe(passage_texts, batch_size=128, n_process=8)

        # Filter passages and extract valid sentences
        for passage_id, passage_text, doc in tqdm(
                zip(passage_ids, passage_texts, docs),
                desc="Filtering passages",
                total=len(passage_ids)
        ):
            valid_sents = [sent.text.strip() for sent in doc.sents if is_valid_sentence(sent)]

            if valid_sents:
                # Keep the original passage text if we found at least one valid sentence
                filtered_passages.append((passage_id, passage_text))
                sentences.extend(valid_sents)
                tsv_writer.writerow([passage_id, passage_text])

    with open(sentences_file, "w", encoding="utf-8") as sf:
        for sentence in sentences:
            sf.write(sentence + "\n")

    print(f"Files '{passages_file}' and '{sentences_file}' have been created.")
    print(f"Total valid passages: {len(filtered_passages)}")
    print(f"Total valid sentences: {len(sentences)}")


# Example call
if __name__ == "__main__":
    generate_files_from_beir("msmarco")
