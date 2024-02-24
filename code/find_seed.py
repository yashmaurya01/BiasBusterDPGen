import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class FindSeed:
	"""
	The goal is to first create two subsets, one of the first category and one of the second category. Figure out who is the minority.
	Embed everything using a sentence encoder. Then for each of the minority find its respective bias-category counterfactual from 
	the majority samples. 

	Now, we'll have some rows in the majority subset, who don't have any minority counterfactuals. These will be considered seeds for 
	synthetic data generation. Only return the most differentiating majority samples of a balancing-count which de-bias the dataset.
	"""
	def __init__(self, bias_category="Gender", category_1_name="male", category_2_name="female"):
		self.bias_category = bias_category
		self.category_1_name = category_1_name
		self.category_2_name = category_2_name
		self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

	def find_most_similar(self, a, b, return_indice=True):
		similarities = cosine_similarity([item['embedding'] for item in a], [item['embedding'] for item in b])
		if return_indice:
			most_similar_indices = np.argmax(similarities, axis=1)
		else:
			most_similar_indices = np.max(similarities, axis=1)
		return most_similar_indices

	def find_seeds(self, df):
		counts = {f'{self.category_1_name}': df[f'{self.category_1_name}'].sum(), f'{self.category_2_name}': df[f'{self.category_2_name}'].sum()}
		majority, minority = sorted(counts, key=counts.get, reverse=True)
		strings = df.row_str.to_list()
		sentence_embeddings = self.model.encode(strings, show_progress_bar=True)
		majority_sentence_embeddings = []
		minority_sentence_embeddings = []
		for index, row in df.iterrows():
			if row[majority]==1:
				majority_sentence_embeddings.append({"row_str": row["row_str"], "embedding": sentence_embeddings[index]})
			elif row[minority]==1:
				minority_sentence_embeddings.append({"row_str": row["row_str"], "embedding": sentence_embeddings[index]})
		similar_majority_indices = self.find_most_similar(minority_sentence_embeddings, majority_sentence_embeddings)
		similar_counterfactuals = []
		minority_counterfactual_embeddings, majority_counterfactual_embeddings = [], []
		for index, minority in enumerate(minority_sentence_embeddings):
			index_of_majority = similar_majority_indices[index]
			similar_counterfactuals.append([minority["row_str"], majority_sentence_embeddings[index_of_majority]["row_str"]])
			minority_counterfactual_embeddings.append(minority["embedding"])
			majority_counterfactual_embeddings.append(majority_sentence_embeddings[index_of_majority]["embedding"])
		similarities_counterfactual_pairs = cosine_similarity(minority_counterfactual_embeddings, majority_counterfactual_embeddings)
		similarities_counterfactual_pairs_scores = np.diag(similarities_counterfactual_pairs)
		indices_of_best_counterfactual_scores = np.argsort(similarities_counterfactual_pairs_scores)[::-1][:10]
		best_counterfactuals = [similar_counterfactuals[idx] for idx in indices_of_best_counterfactual_scores]
		differentiating_majority_scores = self.find_most_similar(majority_sentence_embeddings, minority_sentence_embeddings, False)
		indices_of_seeds = np.argsort(differentiating_majority_scores)[:len(majority_sentence_embeddings)-len(minority_sentence_embeddings)]
		seed_rows_majority_samples = [majority_sentence_embeddings[idx]["row_str"] for idx in indices_of_seeds]
		return seed_rows_majority_samples, best_counterfactuals