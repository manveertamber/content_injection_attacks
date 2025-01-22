import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name, normalize=True, pooling="cls", instruction="", text_maxlength=512, device='cuda', passage_prefix=''):
        super(EmbeddingModel, self).__init__()
        if pooling == "LLM":
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.normalize = normalize
        self.pooling = pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_maxlength = text_maxlength
        self.instruction = instruction
        self.passage_prefix = passage_prefix
        self.device = device
        self.model.gradient_checkpointing_enable()
        if (len(self.instruction) != 0) and (self.instruction[-1] != ' '):
            self.instruction = self.instruction + ' '

    def forward(self, ids, mask):
        model_output = self.model(ids, mask)
        if self.pooling == "mean":
            embs = self.average_pool(model_output.last_hidden_state, mask)  # Mean pooling
        elif self.pooling == "LLM":
            embs = self.last_token_pool(model_output.last_hidden_state, mask)
        else:
            assert(self.pooling == "cls")
            embs = model_output[0][:, 0]  # CLS token representation
        
        if self.normalize:
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        
        return embs

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
   
    def encode_texts(self, texts):
        p = self.tokenizer.batch_encode_plus(
            [(self.passage_prefix + text) for text in texts],
            max_length=self.text_maxlength,
            padding=True,
            return_tensors='pt',
            truncation=True
        )
        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            return self.forward(p['input_ids'].to(self.device), p['attention_mask'].bool().to(self.device))

    def encode_queries(self, queries):
        p = self.tokenizer.batch_encode_plus(
            [(self.instruction + query) for query in queries],
            max_length=self.text_maxlength,
            padding=True,
            return_tensors='pt',
            truncation=True
        )
        with torch.amp.autocast(self.device, dtype=torch.bfloat16):
            return self.forward(p['input_ids'].to(self.device), p['attention_mask'].bool().to(self.device))

