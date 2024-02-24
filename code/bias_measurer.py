import pandas as pd
from openai import OpenAI
import json
import re

class BiasMeasure:
	def __init__(self, openai_api_key, bias_category="Gender", category_1_name="male", category_2_name="female"):
		self.client = OpenAI(api_key=openai_api_key)
		self.category_1_name = category_1_name
		self.category_2_name = category_2_name
		self.BASE_PROMPT = f"""
So here are the first 3 rows of a CSV file, our client is analyzing. We want to use RegEx, to find out whether the dataset is {bias_category} biased, so each row of the CSV file will be treated as a string and you've to create two regex queries, one which represents category 1: {category_1_name} and the other category 2: {category_2_name}. Return two regex queries and ensure, there's no sequences which will fall in both. Based on the CSV record, if say there is a column called "Gender" or "Sex", when discussing Gender Bias, or any other column which directly represents the {bias_category}, then mention the identities from it, else if its pure string or no such column is detected use other strings like pronouns, etc then your regex should include a space before and after the pronouns. However, remember that if its simply a columnar data then it'll not have these spaces, so in those cases do not include space.
		"""
		self.function_template = [
			{
				"name": "regex_gen",
				"description": f"You have to return two RegEx strings, one to represent {category_1_name} and the other to represent {category_2_name}. These regex strings/sequences have to be based on the CSV record, if say there is a column called \"Gender\" or \"Sex\", when discussing Gender Bias, or any other column which directly represents the {bias_category}, then mention the identities from it, else if its pure string or no such column is detected use other strings like pronouns, etc. then your regex should include a space before and after the pronouns. However, remember that if its simply a columnar data then it'll not have these spaces, so in those cases do not include space. Ensure no sequence is common between the two regex.",
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
		# print(f'Dataframe: {df_str}')
		messages = [
			{"role": "system", "content": self.BASE_PROMPT},
			{"role": "user", "content": "These are the first three rows of the CSV:\n\n\"\"\"\n"+df_str+"\n\"\"\""}]
		response = self.client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages, functions=self.function_template, function_call={"name": "regex_gen"}, temperature=0)
		answer = response.choices[0].message.function_call.arguments
		generated_response = json.loads(answer)
		# print(f"Generated response: {generated_response}")
		# Function to remove escape sequences from strings
		def remove_escape_sequences(text):
			return text.replace('\x08', '')


		# Remove escape sequences from the response
		clean_response = {key: remove_escape_sequences(value) for key, value in generated_response.items()}
		# print(f"Clean response: {clean_response}")
		return clean_response

	def evaluate_df(self, df, generated_response):
		scores_list = []
		for index, row in df.iterrows():
			# print()
			row_str = ', '.join(str(value) for value in row.values)
			# print(row_str)
			row_scores = {'regex_category_1': 0, 'regex_category_2': 0}
			for key, regex_pattern in generated_response.items():
				row_scores[key] = len(re.findall(regex_pattern, row_str, re.IGNORECASE))
			# print(row_scores)
			if row_scores['regex_category_1']>row_scores['regex_category_2']:
				row_scores['regex_category_1'] = 1
				row_scores['regex_category_2'] = 0
			elif row_scores['regex_category_2']>row_scores['regex_category_1']:
				row_scores['regex_category_1'] = 0
				row_scores['regex_category_2'] = 1
			else:
				row_scores['regex_category_1'] = 0
				row_scores['regex_category_2'] = 0
			scores_list.append(row_scores)
		scores_df = pd.DataFrame(scores_list)
		scores_df.rename(columns={'regex_category_1': f'{self.category_1_name}', 'regex_category_2': f'{self.category_2_name}'}, inplace=True)
		print(scores_df.describe())

		