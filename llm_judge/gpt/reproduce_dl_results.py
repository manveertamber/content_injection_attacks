'''
DL19

            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         2556         1760          812           30
Expected 1          167          627          730           77
Expected 2           80          424         1128          172
Expected 3          120           72          337          168
Accuracy: 0.7467602591792657
Precision (label=0): 0.8801240096451947
Precision (label=1): 0.5225825130283729
Mean Absolute Error (MAE) for integer scores: 0.6611231101511879

DL20
Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         6725         4183         1949           81
Expected 1          529         1290         1558          164
Expected 2          153          689         1730          252
Expected 3          194          206          685          258
Accuracy: 0.7581129516613387
Precision (label=0): 0.911088839573341
Precision (label=1): 0.4380709899655534
Mean Absolute Error (MAE) for integer scores: 0.6618715489683231

'''

from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score, precision_score
import pandas as pd
import json

queries = []
passages = []
expected_scores = []
qids = []
pids = []

for dataset in ['dl19', 'dl20']:
    with open(f'../dl_qrels/{dataset}.tsv', 'r', encoding='utf-8') as input_pairs:
        for line in input_pairs:
            qid, query, pid, passage, ref_score = line.split('\t')
            qid, query, pid, passage, ref_score = qid.strip(), query.strip(), pid.strip(), passage.strip(), ref_score.strip()

            expected_scores.append(int(ref_score))
            queries.append(query)
            passages.append(passage[:2048])
            qids.append(qid)
            pids.append(pid)
    
    generated_scores = []
    with open(f'gpt_output/gpt4o_batch_output_{dataset}_scores.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            generated_score = json.loads(line)["content"]
            try:
                generated_score = int(generated_score)
            except:
                generated_score = 0
            generated_scores.append(generated_score)  

    conf_matrix = confusion_matrix(expected_scores, generated_scores)

    confusion_df = pd.DataFrame(
        conf_matrix,
        index=[f"Expected {i}" for i in range(conf_matrix.shape[0])],
        columns=[f"Generated {i}" for i in range(conf_matrix.shape[1])]
    )

    print("Confusion Matrix :")
    print(confusion_df)

    binary_expected_scores = [1 if score >= 2 else 0 for score in expected_scores]
    binary_generated_scores = [1 if score >= 2 else 0 for score in generated_scores]

    accuracy = accuracy_score(binary_expected_scores, binary_generated_scores)

    precision_label_0 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=0)
    precision_label_1 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=1)

    mae = mean_absolute_error(expected_scores, generated_scores)

    print(f"Accuracy: {accuracy}")
    print(f"Precision (label=0): {precision_label_0}")
    print(f"Precision (label=1): {precision_label_1}")
    print(f"Mean Absolute Error (MAE) for integer scores: {mae}")

    with open(f'gpt4o_qrel_scores_{dataset}.tsv', 'w', encoding='utf-8') as output_f:
        for i in range(len(expected_scores)):
            expected_score = expected_scores[i]
            output_f.write(qids[i] + '\t' + queries[i] + '\t' + pids[i] + '\t' + passages[i] + '\t' + str(expected_score) + '\t' + str(generated_scores[i]) +'\n')

