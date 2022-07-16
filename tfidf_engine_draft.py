import os
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import pandas as pd
from texts_processing import TextsTokenizer
from gensim.matutils import corpus2csc
from sklearn.metrics.pairwise import cosine_similarity


"""Data loading"""
etalons_df = pd.read_csv(os.path.join("data", "fa_pbid9.csv"))
test_df = pd.read_csv(os.path.join("data", "queries_chat_testing.csv"), sep="\t")
test_texts = list(test_df["query"])

"""Model creating"""
tokenizer = TextsTokenizer()
etalons_texts = list(etalons_df["texts"])
dataset = tokenizer(etalons_texts)
dct = Dictionary(dataset)  # fit dictionary
corpus = [dct.doc2bow(line) for line in dataset]  # convert corpus to BoW format
model = TfidfModel(corpus)  # fit model
vectors = model[corpus]
csc_matrix = corpus2csc(vectors)

results = []
for test_tx in test_texts:
    try:
        test_tokens = tokenizer([test_tx])
        test_corp = dct.doc2bow(test_tokens[0])
        test_vector = model[test_corp]
        test_csc_vector = corpus2csc([test_vector], num_terms=len(dct))
        matrix_scores = cosine_similarity(test_csc_vector.T, csc_matrix.T, dense_output=False)
        i_scs = [(i, sc) for i, sc in zip(matrix_scores.indices, matrix_scores.data)]
        search_result = sorted(i_scs, key=lambda x: x[1], reverse=True)
        # print("searched_text:", test_tx)
        # print("found_text:", etalons_df._get_value(search_result[0][0], "texts"), "score:", search_result[0][1])
        results.append((test_tx, etalons_df._get_value(search_result[0][0], "texts"),
                        etalons_df._get_value(search_result[0][0], "answer_id"), search_result[0][1]))
    except:
        print(test_tx, "\n", search_result)


results_df = pd.DataFrame(results, columns=["searched_text", "found_text", "answer_id", "score"])
results_df.to_csv(os.path.join("data", "queries_chat_testing_pbid9_1.csv"), index=False)