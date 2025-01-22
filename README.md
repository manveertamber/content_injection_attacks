> This repository contains code and scripts used for studying content injection attacks. 

- Please email mtamber@uwaterloo.ca for any questions about this work!
- We will continue to refine this codebase

### Files
- **adversarial_passage_generator.py**  
  - This script defines an AdversarialGenerator class, which manipulates text passages by injecting queries, keywords, or sentences to create adversarial passages for testing

- **get_passages_and_sentences_from_beir_corpora.py**  
  - This script extracts valid sentences from passages from BEIR corpora using heuristics to ensure sentences are meaningful and filters passages without valid sentences

### Directories

1. **attack_passages/**  
   - Contains scripts to produce adversarial passages for particular models and attack settings

2. **attack_results/**  
   - Stores results from model outputs and contains code to evaluate results

3. **classifier/**  
   - Code related to training and evaluating a classifier to detect adversarial passages

4. **embedding_retrieval/**  
   - Code related to training and evaluating embedding models for retrieval

5. **llm_judge/**  
   - Contains scripts to produce and evaluate LLM relevance judgements

6. **model_scores_and_judgements/**  
   - Holds model retrieval results and LLM relevance judgements

7. **queries/**  
   - Contains queries for the studied datasets

8. **random_sentences/**  
   - Holds random sentences used for sentence injection

9. **rel_passage_gen/**  
   - Used to generate relevant passages for particular queries

10. **reranker/**  
    - Code related to evaluating rerankers

