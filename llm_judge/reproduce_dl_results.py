'''

DL19 LLaMa
Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         1734         1485         1822          117
Expected 1          108          266         1135           92
Expected 2           54          122         1535           93
Expected 3           95           50          505           47
Accuracy: 0.6234341252699784
Precision (label=0): 0.9179867143587123
Precision (label=1): 0.40778151889263
Mean Absolute Error (MAE) for integer scores: 0.8768898488120951
Kendall's tau: 0.38920234542555104 (p-value: 0.0)

DL20
Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         3009         1974         2600          197
Expected 1          222          291         1349           78
Expected 2           36           88          851           45
Expected 3           48           81          478           39
Accuracy: 0.6067978218865273
Precision (label=0): 0.9559923464950426
Precision (label=1): 0.2506652474720596
Mean Absolute Error (MAE) for integer scores: 0.9205164236782013
Kendall's tau: 0.3541221987984646 (p-value: 0.0)


DL19 LLaMA adversarial

Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0           11         2024         3123            0
Expected 1            0          134         1467            0
Expected 2            0           62         1742            0
Expected 3            0          115          580            2
Accuracy: 0.48520518358531317
Precision (label=0): 0.9245524296675192
Precision (label=1): 0.3361295921319063
Mean Absolute Error (MAE) for integer scores: 1.1456803455723543
Kendall's tau: 0.3207062149024045 (p-value: 1.9499292609171193e-237)

DL20 LLaMA adversarial
Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0            4         3215         4558            3
Expected 1            0          193         1747            0
Expected 2            0           25          995            0
Expected 3            0           43          602            1
Accuracy: 0.4400140523449851
Precision (label=0): 0.9804597701149426
Precision (label=1): 0.20212496837844676
Mean Absolute Error (MAE) for integer scores: 1.299841911118918
Kendall's tau: 0.32817706427454574 (p-value: 3.851972985186233e-294)


'''


from llm_judge import LLMJudge
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score, precision_score
import pandas as pd
from scipy.stats import kendalltau

for dataset in ['dl19', 'dl20']:

    queries = []
    passages = []
    expected_scores = []
    qids = []
    pids = []

    with open(f'dl_qrels/{dataset}.tsv', 'r', encoding='utf-8') as input_pairs:
        for line in input_pairs:
            qid, query, pid, passage, ref_score = line.split('\t')
            qid, query, pid, passage, ref_score = qid.strip(), query.strip(), pid.strip(), passage.strip(), ref_score.strip()

            expected_scores.append(int(ref_score))
            queries.append(query)
            passages.append(passage[:2048])
            qids.append(qid)
            pids.append(pid)
        
    judge = LLMJudge(prompt_levels=4, adversarial=False)
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

    binary_expected_scores = [1 if score >= 2 else 0 for score in expected_scores]
    binary_generated_scores = [1 if score >= 2 else 0 for score in generated_scores]

    accuracy = accuracy_score(binary_expected_scores, binary_generated_scores)

    precision_label_0 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=0)
    precision_label_1 = precision_score(binary_expected_scores, binary_generated_scores, pos_label=1)

    mae = mean_absolute_error(expected_scores, generated_scores)

    tau, p_val = kendalltau(expected_scores, generated_scores)

    print(f"Accuracy: {accuracy}")
    print(f"Precision (label=0): {precision_label_0}")
    print(f"Precision (label=1): {precision_label_1}")
    print(f"Mean Absolute Error (MAE) for integer scores: {mae}")
    print(f"Kendall's tau: {tau} (p-value: {p_val})\n")

    with open(f'LLM_qrel_scores_{dataset}_basic.tsv', 'w', encoding='utf-8') as output_f:
        for i in range(len(expected_scores)):
            expected_score = expected_scores[i]
            output_f.write(qids[i] + '\t' + queries[i] + '\t' + pids[i] + '\t' + passages[i] + '\t' + str(expected_score) + '\t' + str(generated_scores[i]) +'\n')

    del judge