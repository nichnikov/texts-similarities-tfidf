import os
import json
import pandas as pd

def add_tokens(tokenizer, queries: [()]):
    """"""
    q_i, a_i, m_i, cls, p_i = zip(*queries)
    tokens = tokenizer(cls)
    return list(zip(q_i, a_i, m_i, cls, p_i, tokens))


PATH = r"/home/an/Data/Dropbox/data/fast_answers" # Home
# PATH = r"/home/alexey/Data/Dropbox/data/fast_answers" # Office
# file_name = "data_all_with_locale.json"
# file_name = "qa-full-ru.json"
# file_name = "qa-full-ua.json"
file_name = "qa.json"

with open(os.path.join(PATH, file_name), "r") as f:
    initial_data_json = json.load(f)

initial_data = initial_data_json["data"]

for d in initial_data[:10]:
    print(d)

queries_in = []
for d in initial_data:
    if 9 in d["pubIds"]:
        queries_in += [(d["id"], d["moduleId"], tx, d["pubIds"]) for tx in d["clusters"]]

"""
for i in queries_in:
    print(i)"""

print(len(queries_in))

queries_in_df = pd.DataFrame(queries_in, columns=["answer_id", "moduleId", "texts", "pubIds"])
print(queries_in_df)
queries_in_df.to_csv(os.path.join("data", "fa_pbid9.csv"), index=False)

# print("queries_in:\n", queries_in[:5])

