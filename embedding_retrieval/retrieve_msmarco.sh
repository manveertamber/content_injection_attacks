for model in 'adversarial-random-1.0-retriever' 'adversarial-random-1.1-retriever' 'adversarial-targeted-1.0-retriever'; do       
    python3 -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto --l2-norm --encoder models/${model} \
    --index attack_indices/models_${model}_msmarco_index \
    --topics dl19-passage \
    --output run.beir.adversarial.${model}_dl19.txt \
    --query-prefix "Represent this sentence for searching relevant passages: " \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl19-passage \
    run.beir.adversarial.${model}_dl19.txt
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage \
    run.beir.adversarial.${model}_dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl19-passage \
    run.beir.adversarial.${model}_dl19.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl19-passage \
    run.beir.adversarial.${model}_dl19.txt

    python3 -m pyserini.search.faiss \
    --threads 16 --batch-size 512 \
    --encoder-class auto --l2-norm --encoder models/${model} \
    --index attack_indices/models_${model}_msmarco_index \
    --topics dl20 \
    --output run.beir.adversarial.${model}_dl20.txt \
    --query-prefix "Represent this sentence for searching relevant passages: " \
    --hits 1000

    python -m pyserini.eval.trec_eval -c -l 2 -m map dl20-passage \
    run.beir.adversarial.${model}_dl20.txt
    python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl20-passage \
    run.beir.adversarial.${model}_dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 dl20-passage \
    run.beir.adversarial.${model}_dl20.txt
    python -m pyserini.eval.trec_eval -c -l 2 -m recall.1000 dl20-passage \
    run.beir.adversarial.${model}_dl20.txt
done