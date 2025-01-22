import json
from tqdm import tqdm

from openai import OpenAI

client = OpenAI(
  api_key='',
  project='',
)

word_len = 200

datasets = ['dl19', 'dl20']

for dataset in datasets:
    qids = []
    queries = []

    with open(f'../queries/{dataset}-queries.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split('\t')
            qid = vals[0].strip()
            query = vals[1].strip()
            qids.append(qid)
            queries.append(query)

    system_prompt = 'You are a helpful assistant who writes perfectly relevant passages when given queries. Follow instructions precisely.'
    user_prompt = '''Instructions: Carefully review the following query and write a passage that fully and accurately addresses it. The passage should be approximately '''+ str(word_len) + ''' words, concise, and relevant.

Respond with the plain text passage only.

Query: {}
Passage:
'''

    assert(len(qids) == len(queries))

    print(dataset)
    with open(f'generated_passages/{dataset}_generated_passages_{word_len}_words.tsv', 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(queries))):
            query = queries[i]
            llm_input = user_prompt.format(query)

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": llm_input}
                    ],
                    max_tokens=512,
                    #temperature=1.0
                )

                print(response.choices[0].message.content)
                generated_text = response.choices[0].message.content

                generated_text = generated_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').strip()
                if (generated_text.startswith('"')) and (generated_text.endswith('"')):
                    generated_text = generated_text[1:-1]

                f.write(qids[i] + '\t' + generated_text + '\n')

            except Exception as e:
                print(f"Error generating passage for query {qids[i]}: {e}")
                f.write(qids[i] + '\t' + "ERROR: Could not generate passage\n")
