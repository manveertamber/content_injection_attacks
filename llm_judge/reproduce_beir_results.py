'''
nq
            Generated 0  Generated 1
Expected 0            0            0
Expected 1          570         3631
Accuracy: 0.8643180195191621
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.1356819804808379

hotpotqa
            Generated 0  Generated 1
Expected 0            0            0
Expected 1         5517         9293
Accuracy: 0.6274814314652262
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.3725185685347738

fiqa
            Generated 0  Generated 1
Expected 0            0            0
Expected 1          476         1230
Accuracy: 0.7209847596717468
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.2790152403282532

scifact
            Generated 0  Generated 1
Expected 0            0            0
Expected 1           91          248
Accuracy: 0.7315634218289085
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.26843657817109146

trec-covid
            Generated 0  Generated 1  Generated 2
Expected 0        28383         9325         3955
Expected 1         3632         3984         2840
Expected 2         2613         4771         6833
Accuracy: 0.7056650988904969
Precision (label=0): 0.8196546147626198
Precision (label=1): 0.5811782515453513
Mean Absolute Error (MAE) for integer scores: 0.5080800771828268

nfcorpus
            Generated 0  Generated 1  Generated 2
Expected 0            0            0            0
Expected 1         8647         1584         1527
Expected 2          220          134          222
Accuracy: 0.2810929138965461
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.87141235608886

dbpedia-entity
            Generated 0  Generated 1  Generated 2
Expected 0        23125         3661         1443
Expected 1         3464         2699         2622
Expected 2         2368         1760         2373
Accuracy: 0.7486843617143514
Precision (label=0): 0.798597921055358
Precision (label=1): 0.649402390438247
Mean Absolute Error (MAE) for integer scores: 0.4395955417672067

fever
            Generated 0  Generated 1
Expected 0            0            0
Expected 1         2368         5569
Accuracy: 0.7016504976691446
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.2983495023308555

climate-fever
            Generated 0  Generated 1
Expected 0            0            0
Expected 1         2704         1977
Accuracy: 0.42234565263832513
Precision (label=0): 0.0
Precision (label=1): 1.0
Mean Absolute Error (MAE) for integer scores: 0.5776543473616749
'''



from llm_judge import LLMJudge
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score, precision_score
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


datasets = [{"name": 'nq', "relevancy_levels": 2},
            {"name": 'hotpotqa', "relevancy_levels": 2},
            {"name": 'fiqa', "relevancy_levels": 2},
            {"name": 'scifact', "relevancy_levels": 2},
            {"name": 'trec-covid', "relevancy_levels": 3},
            {"name": 'nfcorpus', "relevancy_levels": 3},
            {"name": 'dbpedia-entity', "relevancy_levels": 3},
            {"name": 'fever', "relevancy_levels": 2},
            {"name": 'climate-fever', "relevancy_levels": 2},]

for dataset in datasets:
    print('dataset', dataset)
    queries = []
    passages = []
    expected_scores = []
    qids = []
    pids = []

    name = dataset['name']
    relevancy_levels = dataset['relevancy_levels']
    judge = LLMJudge(prompt_levels=relevancy_levels)

    ds = load_dataset(f"BeIR/{name}", "corpus")['corpus']
    id_key = '_id'
    passages_dict = {
        str(line[id_key]): str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        for line in tqdm(ds, desc="Loading corpus")
    }

    ds = load_dataset(f"BeIR/{name}", "queries")['queries']
    queries_dict = {
        str(line[id_key]): str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        for line in tqdm(ds, desc="Loading queries")
    }

    qrels_dict = {}
    ds = load_dataset(f"BeIR/{name}-qrels")
    if 'test' in ds:
        ds = ds['test']
    else:
        ds = ds['validation']
    for line in tqdm(ds, desc="Loading qrels"):
        if str(line['query-id']) in qrels_dict:
            qrels_dict[str(line['query-id'])][str(line['corpus-id'])] = line['score']
        else:
            qrels_dict[str(line['query-id'])] = {str(line['corpus-id']): line['score']}

    for qid in qrels_dict.keys():
        for pid in qrels_dict[qid].keys():
            score = int(qrels_dict[qid][pid])
            if score < 0:
                score = 0

            expected_scores.append(int(score))
            queries.append(queries_dict[qid])
            passages.append(passages_dict[pid])
            qids.append(qid)
            pids.append(pid)

    prompts = judge.generate_prompts_from_pairs(queries, passages)
    generated_scores = judge.generate_scores_from_prompts(prompts)

    conf_matrix = confusion_matrix(expected_scores, generated_scores)

    confusion_df = pd.DataFrame(
        conf_matrix,
        index=[f"Expected {i}" for i in range(conf_matrix.shape[0])],
        columns=[f"Generated {i}" for i in range(conf_matrix.shape[1])]
    )

    print("Confusion Matrix :")
    print(confusion_df)

    binary_expected_scores = [1 if score >= 1 else 0 for score in expected_scores]
    binary_generated_scores = [1 if score >= 1 else 0 for score in generated_scores]

    accuracy = accuracy_score(binary_expected_scores, binary_generated_scores)

    precision_label_0 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=0)
    precision_label_1 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=1)

    mae = mean_absolute_error(expected_scores, generated_scores)

    print(f"Accuracy: {accuracy}")
    print(f"Precision (label=0): {precision_label_0}")
    print(f"Precision (label=1): {precision_label_1}")
    print(f"Mean Absolute Error (MAE) for integer scores: {mae}")


    with open(f'LLM_qrel_scores_{name}.tsv', 'w', encoding='utf-8') as output_f:
        for i in range(len(expected_scores)):
            expected_score = expected_scores[i]
            output_f.write(qids[i] + '\t' + queries[i] + '\t' + pids[i] + '\t' + passages[i] + '\t' + str(expected_score) + '\t' + str(generated_scores[i]) +'\n')
    
    del judge