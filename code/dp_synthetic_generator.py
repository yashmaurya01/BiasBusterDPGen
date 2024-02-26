import pandas as pd
from openai import OpenAI
import numpy as np

class DPPromptSyntheticGenerator:
	def __init__(self, openai_api_key, private_df, batch_size=4):
		self.private_df = private_df
		self.batch_size = batch_size
		self.client = OpenAI(api_key=openai_api_key)
		self.PROMPT_TEMPLATE = f"""
You are a chatbot which is to create synthetic data generation by observing a few samples from the original private dataset. As usual you've to incorporate most, if not all, similarity/context. However, it shouldn't be an exact clone. Find, where modifications can be made and make appropriate changes.
Observe the following data points and figure out the structure of the dataset, focus on the '"' and ','. Generate a sample which is similar and inspired by the given data points.
		"""

	def generate_next_logprob(self, samples):
		messages = [
			{"role": "system", "content": self.PROMPT_TEMPLATE},
			{"role": "user", "content": "Here are some samples from the private dataset:\n\n\"\"\"\n"+samples+"\n\"\"\"\n\nSynethetic Sample:\n"}
		]
		response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages, temperature=0, n=1, max_tokens=1, top_logprobs=5, logprobs=True)
		response = response.choices[0].logprobs.content[0].top_logprobs
		options = []
		for idx, r in enumerate(response):
			token = r.token
			if idx==0 and not any(char.isalpha() for char in token):
				return token
			prob = r.logprob
			options.append([token, prob])
		

	def generate_new_point(self):
		batch_df = self.private_df.sample(n=self.batch_size)
		batch_txt = ""
		for index, row in batch_df.iterrows():
			row_str = '"'+'", "'.join(str(value) for value in row.values)+'"\n'
			batch_txt += row_str
		self.generate_next_logprob(batch_txt)