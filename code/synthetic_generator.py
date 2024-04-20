import pandas as pd
from openai import OpenAI
import numpy as np

class PromptSyntheticGenerator:
	def __init__(self, openai_api_key, minority_category="female", majority_category="male"):
		self.client = OpenAI(api_key=openai_api_key)
		self.PROMPT_TEMPLATE = f"""
Observe the following CSV row, observe the , and the " and figure out the structure. Generate a counterfactual for the following sample, wherein you replace {majority_category} entity to {minority_category} entity. Carefully consider the give CSV row and then switch the least number of terms/words to make it switch to its counterfactual version.
		"""

	def generate_synthetic(self, row):
		messages = [
			{"role": "system", "content": self.PROMPT_TEMPLATE},
			{"role": "user", "content": "Here is the row:\n\n\"\"\"\n"+row+"\n\"\"\"\n\nSynethetic Counterfactual Sample:"}
		]
		response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages, temperature=0)
		answer = response.choices[0].message.content
		return answer