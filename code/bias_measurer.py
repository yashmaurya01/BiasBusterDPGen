import pandas as pd

class BiasMeasure:
	def __init__(self, bias_category="Gender", category_1_name="male", category_2_name="female"):
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
		df_str = df.head(3).to_string()
		