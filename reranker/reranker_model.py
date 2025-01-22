import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, T5ForConditionalGeneration
from tqdm import tqdm

class RerankerModel(torch.nn.Module):
    def __init__(self, model_name, text_maxlength=512, batch_size=32, device='cuda'):
        super(RerankerModel, self).__init__()
        self.device = device

        if 'rankt5' in model_name.lower():
            self.model_type = 'rankt5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'monot5' in model_name.lower():
            self.model_type = 'monot5'
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif 'minilm' in model_name.lower():
            self.model_type = 'sentence_transformer'
            self.model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
            if 'reranker_model_listwise' in model_name:
                state_dict = torch.load(model_name)
                for key in list(state_dict.keys()):
                    if 'model.' in key:
                        state_dict[key.replace('model.', '')] = state_dict[key]
                        del state_dict[key]
                self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

        self.batch_size = batch_size
        self.text_maxlength = text_maxlength

    def forward(self, input_tensors):
        if self.model_type == 'monot5':
            output = self.model.generate(**input_tensors, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores = torch.stack(output.scores)
            scores = torch.nn.functional.log_softmax(scores[0][:, [1176, 6136]], dim=1)[:, 0].tolist()
        elif self.model_type == 'rankt5':
            output = self.model.generate(**input_tensors, max_length=2, return_dict_in_generate=True, output_scores=True)
            scores = torch.stack(output.scores)
            scores = scores[0][:, 32089].tolist() # <extra_id_10>
        else:
            scores = self.model(**input_tensors).logits.squeeze(-1).tolist()

        return scores
    
    def score_pairs(self, pairs):
        '''
        pairs: list of pairs [[q1, p1], [q2, p2] ... ]
        '''

        if self.model_type == 'monot5':
            input_texts = [f"Query: {pair[0]} Document: {pair[1]} Relevant:" for pair in pairs]
            input_tensors = self.tokenizer(input_texts, return_tensors='pt', padding=True, max_length=self.text_maxlength, truncation=True).to(self.device)
        elif self.model_type == 'rankt5':
            input_texts = [f"Query: {pair[0]} Document: {pair[1]}" for pair in pairs]
            input_tensors = self.tokenizer(input_texts, return_tensors='pt', padding=True, max_length=self.text_maxlength, truncation=True).to(self.device)
        else:
            input_tensors = self.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs], return_tensors='pt', padding=True, max_length=self.text_maxlength, truncation=True).to(self.device)
        
        scores = []
        
        with torch.no_grad():
            with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                for i in tqdm(range(0, len(input_tensors['input_ids']), self.batch_size)):
                    batch = {k: v[i:i + self.batch_size] for k, v in input_tensors.items()}
                    scores.extend(self.forward(batch))
        
        return scores

  
# --- TESTS ---
def test_reranker_model():
    sample_pairs = [
        ["How many people live in Berlin?", "Berlin has a population of 3,520,031 registered inhabitants."],
        ["What is the capital of Germany?", "Berlin is the capital of Germany."],
        ["What is the capital of Germany?", "A dog is the best kind of pet"]
    ]

    monot5_model = RerankerModel("castorini/monot5-base-msmarco-10k", text_maxlength=128, batch_size=32, device="cuda").eval()
    rankt5_model = RerankerModel("Soyoung97/RankT5-base", text_maxlength=128, batch_size=32, device="cuda").eval()
    sentence_transformer_model = RerankerModel("cross-encoder/ms-marco-MiniLM-L-12-v2", text_maxlength=128, batch_size=32, device="cuda").eval()

    # Test monot5
    scores_monot5 = monot5_model.score_pairs(sample_pairs)
    assert isinstance(scores_monot5, list), "monot5 output should be a list"
    assert len(scores_monot5) == len(sample_pairs), "monot5 output length should match input pairs"
    print('scores_monot5', scores_monot5)

    # Test rankt5
    scores_rankt5 = rankt5_model.score_pairs(sample_pairs)
    assert isinstance(scores_rankt5, list), "rankt5 output should be a list"
    assert len(scores_rankt5) == len(sample_pairs), "rankt5 output length should match input pairs"
    print('scores_rankt5', scores_rankt5)

    # Test sentence transformer
    scores_sentence_transformer = sentence_transformer_model.score_pairs(sample_pairs)
    assert isinstance(scores_sentence_transformer, list), "sentence_transformer output should be a list"
    assert len(scores_sentence_transformer) == len(sample_pairs), "sentence_transformer output length should match input pairs"
    print('scores_sentence_transformer', scores_sentence_transformer)

    print("All tests passed!")

if __name__ == "__main__":
    test_reranker_model()