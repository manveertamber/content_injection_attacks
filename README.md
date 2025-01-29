# Content Injection Attacks in Neural IR Models

This repository contains code and scripts for studying *content injection attacks* in neural information retrieval models. Content injection attacks involve inserting query and query terms into non-relevant passages to make them seem relevant and inserting non-relevant or even harmful text into seemingly relevant passages to promote misleading or malicious content in search.

We will continue to refine this codebase. For questions or support, please reach out to [mtamber@uwaterloo.ca](mailto:mtamber@uwaterloo.ca).

---

## Overview
This repository offers:
- Scripts to generate adversarial passages and evaluate model vulnerability
- Scripts to train and evaluate classifiers and embedding models more robust to content injection attacks
---

## Repository Structure

### Directories

1. **attack_passages/**  
   Scripts for creating adversarial passages tailored to specific models and attack scenarios.

2. **attack_results/**  
   Stores model outputs under attack, along with evaluation scripts to analyze vulnerability.

3. **classifier/**  
   Scripts for training and testing a classifier that flags adversarially modified passages.

4. **embedding_retrieval/**  
   Scripts to fine-tune and evaluate embedding models for passage retrieval that are more robust to adversarial passages.

5. **llm_judge/**  
   Scripts to evaluate large language model (LLM) judgments of passage relevance.

6. **model_scores_and_judgements/**  
   Retrieval results and LLM-based relevance judgments for different experiments.

7. **queries/**  
   Query sets for the datasets used in these experiments.

8. **random_sentences/**  
   Random sentences used for sentence injection.

9. **rel_passage_gen/**  
   Scripts to produce relevant passages given specific queries.

10. **reranker/**  
   Scripts for evaluating rerankers.

### Files

- **adversarial_passage_generator.py**  
  Implements an `AdversarialGenerator` class to inject queries, keywords, or arbitrary text into passages, creating adversarial passages for testing.

- **get_passages_and_sentences_from_beir_corpora.py**  
  Extracts valid sentences from BEIR corpora passages using heuristics and filters out passages lacking meaningful sentences.

---

