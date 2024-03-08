import pandas as pd
from openai import OpenAI
from tqdm import trange
from scipy.special import softmax
from diffprivlib.mechanisms import Exponential
from diffprivlib.accountant import BudgetAccountant
from diffprivlib.tools import median
import numpy as np
import time

class DPPromptSyntheticGenerator:
	def __init__(self, openai_api_key, private_df, batch_size=4, epsilon=10000):
		self.private_df = private_df
		self.batch_size = batch_size
		self.epsilon = epsilon
		self.client = OpenAI(api_key=openai_api_key)
		self.PROMPT_TEMPLATE = f"""
You are a chatbot which is to create synthetic data generation by observing a few samples from the original private dataset. As usual you've to incorporate most, if not all, similarity/context. However, it shouldn't be an exact clone. Find, where modifications can be made and make appropriate changes.
Observe the following data points and figure out the structure of the dataset, focus on the '"' and ','. Generate a sample which is similar and inspired by the given data points.
		"""

	def generate_next_logprob(self, samples, generated_till_now=None, quote_in=False):
		samples = samples.replace('"', '')
		time.sleep(3)
		if generated_till_now is None:
			print(samples)
			messages = [
				{"role": "system", "content": self.PROMPT_TEMPLATE},
				{"role": "user", "content": "Here are some samples from the private dataset:\n\n\"\"\"\n"+samples+"\n\"\"\"\n\nSynethetic Sample:\n"}
			]
		else:
			messages = [
				{"role": "system", "content": self.PROMPT_TEMPLATE},
				{"role": "user", "content": "Here are some samples from the private dataset:\n\n\"\"\"\n"+samples+"\n\"\"\"\n\nSynethetic Sample:\n"},
				{"role": "system", "content": "Remember the samples provided, follow the structure given. No double quotes or incorrect ',' placements. Do not repeat '\"', you should never generate a '\"\"' or a '\"\"\"'. Ensure that you will continue from where you left off in your response after your last line as if nothing happened, ensuring you will not repeat anything, here is the rest of the response you've generated till now! Do not repeat \" or , if you previously mentioned them. Remember whatever you output will be directly appended to the previous string, so if the previous string ends with \" do not begin your next text with it. Understand that you're continuing the previous string as it is. You are not creating your own stuff, use as few \" as you can. Ensure there's no repeating \" \" spaces. Be concise."},
				{"role": "assistant", "content": str(generated_till_now)}
			]
		if quote_in:
			response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", seed=0, messages=messages, temperature=1, n=1, max_tokens=1, top_logprobs=5, logprobs=True, logit_bias={'34':-100})
		else:
			response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", seed=0, messages=messages, temperature=1, n=1, max_tokens=1, top_logprobs=5, logprobs=True)
		stop_reason = str(response.choices[0].finish_reason)
		response = response.choices[0].logprobs.content[0].top_logprobs
		options = []
		non_alpha_texts = []
		for idx, r in enumerate(response):
			token = r.token
			if not any(char.isalpha() for char in token):
				non_alpha_texts.append(True)
				#return token, stop_reason
			else:
				non_alpha_texts.append(False)
			prob = r.logprob
			options.append([token, prob])
		probability_of_trues = sum(non_alpha_texts) / len(non_alpha_texts)
		if probability_of_trues>0.5:
			return options[0][0], stop_reason
		return options, stop_reason
		
	def generate_new_point(self, self_max_tokens=100):
		acc = BudgetAccountant(epsilon=1e6, delta=0)
		batch_df = self.private_df.sample(n=self.batch_size)
		batch_txt = ""
		for index, row in batch_df.iterrows():
			row_str = '"'+'", "'.join(str(value) for value in row.values)+'"\n'
			batch_txt += row_str
		new_string = ""
		quote_in = False
		for i in trange(self_max_tokens):
			if len(new_string)==0:
				next_out, stop_reason = self.generate_next_logprob(batch_txt, quote_in=quote_in)
			else:
				next_out, stop_reason = self.generate_next_logprob(batch_txt, new_string, quote_in=quote_in)
			if type(next_out) is str:
				new_string += next_out
				if '"' in next_out:
					quote_in=True
				else:
					quote_in = False
			else:
				scores = [item[1] for item in next_out]
				probabilities = softmax(scores)
				exp_mech = Exponential(epsilon=self.epsilon, sensitivity=1.0, utility=probabilities.tolist())
				median(probabilities, bounds=(0, 1), epsilon=self.epsilon, accountant=acc)
				index_of_choice = exp_mech.randomise()
				new_string += next_out[index_of_choice][0]
				if '"' in next_out[index_of_choice][0]:
					quote_in=True
				else:
					quote_in = False
			if stop_reason!="length":
				break
			print("Privacy Accounted:", acc.total())
			print(new_string)
		print("Loop fucking ended")