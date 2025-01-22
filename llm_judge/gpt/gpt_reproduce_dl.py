'''
DL19
Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         3654         1135          262          107
Expected 1          384          772          295          150
Expected 2          152          496          481          675
Expected 3          140          122          117          318
Accuracy: 0.8138228941684665
Precision (label=0): 0.8672501823486506
Precision (label=1): 0.6615384615384615
Mean Absolute Error (MAE) for integer scores: 0.5631749460043196
Kendall's tau: 0.5488157159424697 (p-value: 0.0)

DL20

Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         5913         1391          306          170
Expected 1          472          870          354          244
Expected 2           99          326          295          300
Expected 3           95          107          100          344
Accuracy: 0.8506060073774812
Precision (label=0): 0.9323843416370107
Precision (label=1): 0.4917179365830573
Mean Absolute Error (MAE) for integer scores: 0.46109256982258917
Kendall's tau: 0.5493232967767342 (p-value: 0.0)

DL19 ADV

Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         2429         2309          348           72
Expected 1          111          987          383          120
Expected 2           63          520          660          561
Expected 3           93          156          147          301
Accuracy: 0.8104751619870411
Precision (label=0): 0.8752249550089982
Precision (label=1): 0.6439043209876543
Mean Absolute Error (MAE) for integer scores: 0.6371490280777538
Kendall's tau: 0.5277854349564499 (p-value: 0.0)


DL20 ADV

Confusion Matrix :
            Generated 0  Generated 1  Generated 2  Generated 3
Expected 0         4297         2923          425          135
Expected 1          170         1089          486          195
Expected 2           27          344          407          242
Expected 3           30          157          159          300
Accuracy: 0.8419989460741261
Precision (label=0): 0.9382538453026447
Precision (label=1): 0.4716900808854832
Mean Absolute Error (MAE) for integer scores: 0.5644651326190058
Kendall's tau: 0.5280713316561686 (p-value: 0.0)
'''

import sys
sys.path.append('..')
from llm_judge import LLMJudge
import pandas as pd
import json
from openai import AzureOpenAI
from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score, precision_score
from scipy.stats import kendalltau


from tqdm import tqdm

client = AzureOpenAI(
  azure_endpoint = '', 
  api_key='',  
  api_version="2024-08-01-preview"
)

for dataset in ['dl19', 'dl20']:
    queries = []
    passages = []
    expected_scores = []
    qids = []
    pids = []
    with open(f'../dl_qrels/{dataset}.tsv', 'r', encoding='utf-8') as input_pairs:
        for line in input_pairs:
            qid, query, pid, passage, ref_score = line.split('\t')
            qid, query, pid, passage, ref_score = qid.strip(), query.strip(), pid.strip(), passage.strip(), ref_score.strip()

            expected_scores.append(int(ref_score))
            queries.append(query)
            passages.append(passage[:2048])
            qids.append(qid)
            pids.append(pid)

    judge = LLMJudge(model_name='gpt4o', adversarial=True)
    prompts = judge.generate_prompts_from_pairs(queries, passages)

    generated_scores = []

    for prompt in tqdm(prompts):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        try:
            generated_scores.append(int(response.choices[0].message.content))
        except:
            print("FAILED TO PARSE " + response.choices[0].message.content)
            generated_scores.append(0)


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

    with open(f'gpt4o_qrel_scores_{dataset}_adv.tsv', 'w', encoding='utf-8') as output_f:
        for i in range(len(expected_scores)):
            expected_score = expected_scores[i]
            output_f.write(qids[i] + '\t' + queries[i] + '\t' + pids[i] + '\t' + passages[i] + '\t' + str(expected_score) + '\t' + str(generated_scores[i]) +'\n')

    del judge
        
