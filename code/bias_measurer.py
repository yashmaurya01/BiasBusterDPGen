import pandas as pd
from openai import OpenAI
import json
import re

class BiasMeasure:
	def __init__(self, openai_api_key, bias_category="Gender", category_1_name="male", category_2_name="female"):
		self.client = OpenAI(api_key=openai_api_key)
		self.BASE_PROMPT = f"""
So here are the first 3 rows of a CSV file, our client is analyzing. We want to use RegEx, to find out whether the dataset is {bias_category} biased, so each row of the CSV file will be treated as a string and you've to create two regex queries, one which represents category 1: {category_1_name} and the other category 2: {category_2_name}. Return two regex queries and ensure, there's no sequences which will fall in both. Based on the CSV record, if say there is a column called "Gender" or "Sex", when discussing Gender Bias, or any other column which directly represents the {bias_category}, then mention the identities from it, else if its pure string or no such column is detected use other strings like pronouns, gendered words, etc.
		"""
		self.function_template = [
			{
				"name": "regex_gen",
				"description": f"You have to return two RegEx strings, one to represent {category_1_name} and the other to represent {category_2_name}. These regex strings/sequences have to be based on the CSV record, if say there is a column called \"Gender\" or \"Sex\", when discussing Gender Bias, or any other column which directly represents the {bias_category}, then mention the identities from it, else if its pure string or no such column is detected use other strings like pronouns, gendered words, etc. Ensure no sequence is common between the two regex.",
				"parameters": {
					"type": "object",
					"properties": {
						"regex_category_1": {
							"type": "string",
							"description": f"Contains the RegEx string representing {category_1_name}.",
						},
						"regex_category_2": {
							"type": "string",
							"description": f"Contains the RegEx string representing {category_2_name}.",
						}
					},
				"required": ["regex_category_1", "regex_category_2"] 
				}
			}
		]

	def make_query(self, df):
		df_str = df.head(3).to_string().lower()
		messages = [
			{"role": "system", "content": self.BASE_PROMPT},
			{"role": "user", "content": "These are the first three rows of the CSV:\n\n\"\"\"\n"+df_str+"\n\"\"\""}]
		response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages, functions=self.function_template, function_call={"name": "regex_gen"}, temperature=0)
		answer = response.choices[0].message.function_call.arguments
		generated_response = json.loads(answer)
		return generated_response

	def evaluate_df(self, df, generated_response):
		scores_list = []
		for index, row in df.iterrows():
			row_str = row.to_string()
			row_scores = {'regex_category_1': 0, 'regex_category_2': 0}
			for key, regex_pattern in generated_response.items():
				row_scores[key] = len(re.findall(regex_pattern, row_str, re.IGNORECASE))
			scores_list.append(row_scores)
		scores_df = pd.DataFrame(scores_list)
		print(scores_df.describe())

		