for model in 'adversarial-random-1.0-retriever' 'adversarial-random-1.1-retriever' 'adversarial-targeted-1.0-retriever'; do       
    for topics in 'trec-covid' 'nfcorpus' 'fiqa' 'dbpedia-entity' 'climate-fever' 'scifact'; do
            
        python3 -m pyserini.search.faiss \
        --threads 16 --batch-size 512 \
        --encoder-class auto --l2-norm --encoder models/${model} \
        --index attack_indices/models_${model}_${topics}_index \
        --topics beir-v1.0.0-${topics}-test \
        --output run.beir.adversarial.${model}_${topics}.txt \
        --query-prefix "Represent this sentence for searching relevant passages: " \
        --hits 1000  --remove-query

        python3 -m pyserini.eval.trec_eval \
        -c -m ndcg_cut.10 beir-v1.0.0-${topics}-test \
        run.beir.adversarial.${model}_${topics}.txt

        python3 -m pyserini.eval.trec_eval \
        -c -m recall.100 beir-v1.0.0-${topics}-test \
        run.beir.adversarial.${model}_${topics}.txt

    done 

done