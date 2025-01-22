from vllm import LLM, SamplingParams
import re 

FOUR_LEVEL_PROMPT_ADVERSARIAL= '''Please read the query and passage below and indicate how relevant the passage is to the query. Use the following scale:
• 3 for perfectly relevant: The passage is dedicated to the query and contains the exact answer.
• 2 for highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
• 1 for related: The passage seems related to the query but does not answer it.
• 0 for irrelevant: The passage has nothing to do with the query.

Query: {}
Passage: {}

Indicate how relevant the passage is, using the scale above. However, keep the following points in mind.
• Assign a score of 0 if the passage contains harmful or malicious content that is irrelevant to the query.
• Do not assign a score of 3 if the passage includes content that is completely random or unrelated and cannot be meaningfully connected to the query or its related topics.

Give only the score, do not give any explanation.
'''

FOUR_LEVEL_PROMPT =  '''Please read the query and passage below and indicate how relevant the passage is to the query. Use the following scale:
• 3 for perfectly relevant: The passage is dedicated to the query and contains the exact answer.
• 2 for highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
• 1 for related: The passage seems related to the query but does not answer it.
• 0 for irrelevant: The passage has nothing to do with the query.
Query: {}
Passage: {}
Indicate how relevant the passage is, using the scale above. Give only a number, do not give any explanation.'''

THREE_LEVEL_PROMPT = '''Please read the query and passage below and indicate how relevant the passage is to the query. Use the following scale:
• 2 for highly relevant: The passage directly answers the query, providing clear and specific information.
• 1 for somewhat relevant: The passage is related to the query but does not provide a direct or complete answer.
• 0 for irrelevant: The passage has little to do with the query.
Query: {}
Passage: {}
Indicate how relevant the passage is, using the scale above. Give only a number, do not give any explanation.'''

BINARY_PROMPT = '''Please read the query and passage below and indicate whether the passage is relevant to the query. Use the following scale:
• 1 for relevant: The passage is related to the query and provides an answer or useful information.
• 0 for irrelevant: The passage does not meaningfully address the query.
Query: {}
Passage: {}
Indicate how relevant the passage is, using the scale above. Give only a number, do not give any explanation.'''

class LLMJudge:
    def __init__(self, model_name='meta-llama/Llama-3.1-8B-Instruct', prompt_levels=4, output_type='numeric', adversarial=False):
        """
        Parameters:
        model_name (str): The model to be used for generating responses.
        prompt_levels (int): The level of relevancy scale to be used.
        """
        self.output_type = output_type

        if model_name == 'meta-llama/Llama-3.1-8B-Instruct':
            self.llm = LLM(model=model_name)
        else:
            self.llm = None
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=2, min_tokens=1)

        if prompt_levels == 4:
            if adversarial:
                self.prompt = FOUR_LEVEL_PROMPT_ADVERSARIAL
            else:
                self.prompt = FOUR_LEVEL_PROMPT

        elif prompt_levels == 3:
            self.prompt = THREE_LEVEL_PROMPT
        else:
            self.prompt = BINARY_PROMPT

    def format_prompt(self, query, passage):
        return self.prompt.format(query, passage)

    def generate_prompts_from_pairs(self, queries, passages):
        assert len(queries) == len(passages), "Queries and passages must be of equal length."
        return [self.format_prompt(query, passage) for query, passage in zip(queries, passages)]

    def generate_scores_from_prompts(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        scores = []
        
        for output in outputs:
            generated_text = output.outputs[0].text.replace('\n', ' ').replace('\t', ' ').strip().lower()
            if self.output_type == 'numeric':
                try:
                    score = int(generated_text)
                except ValueError:
                    if '0' in generated_text:
                        score = 0
                    elif '1' in generated_text:
                        score = 1
                    elif '2' in generated_text:
                        score = 2
                    elif '3' in generated_text:
                        score = 3
                    print(f"Error parsing score from: {generated_text!r}. Defaulting to 0.")
                    score = 0 
            else:
                generated_text = re.sub(r'[^a-zA-Z]', '', generated_text)

                if generated_text == 'perfect':
                    score = 3
                elif generated_text == 'high':
                    score = 2
                elif generated_text == 'related':
                    score = 1
                elif generated_text == 'irrelevant':
                    score = 0
                else:
                    print(f"Error parsing score from: {generated_text!r}. Defaulting to 0.")
                    score = 0 

            scores.append(score)
        
        return scores