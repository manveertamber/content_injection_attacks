from datasets import load_dataset
from tqdm import tqdm

datasets = ['fiqa', 'scifact', 'trec-covid', 'nfcorpus', 'climate-fever', 'dbpedia-entity']

for dataset in datasets:
    qids = []
    queries = []
    qrel_queries = set()

    print('dataset', dataset)
    ds = load_dataset("BeIR/" + dataset + "-qrels")['test']; id_key='query-id'
    for line in tqdm(ds):        
        qid = str(line[id_key])
        rel_score = int(line['score'])
        if rel_score > 0:
            qrel_queries.add(qid)
    ds = load_dataset("BeIR/" + dataset, "queries")['queries']; id_key='_id'
    for line in tqdm(ds):        
        qid = str(line[id_key])
        text = str(line['text']).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()[:2048]
        if qid in qrel_queries:
            qids.append(qid)
            queries.append(text)

    with open(f'{dataset}-queries.tsv', 'w', encoding='utf-8') as f:
        for i in range(len(qids)):
            f.write(qids[i] + '\t' + queries[i] + '\n')



