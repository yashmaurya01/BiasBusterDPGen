import pandas as pd
import numpy as np
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
		print(len(majority_sentence_embeddings), len(minority_sentence_embeddings))