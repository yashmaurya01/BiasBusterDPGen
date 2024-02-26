from code import BiasMeasure, FindSeed, PromptSyntheticGenerator, DPPromptSyntheticGenerator
import pandas as pd

with open(".secrets", "r") as f:
	openai_api_key = f.read()

print("Read API Key")
df = pd.read_csv("./data/biased/tabular/bank.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# bm = BiasMeasure(openai_api_key)#,"Maritial Status", "Single", "Married")
# regex_queries = bm.make_query(df)
# print("RegEx Done")
# scores_df = bm.evaluate_df(df, regex_queries)
# print("Score DF Generated")
# fs = FindSeed("Maritial Status")#,"Single", "Married")
# print("Model Loaded")
# seed_rows, example_counterfactuals = fs.find_seeds(scores_df)
# print("Seed Found")
# psg = PromptSyntheticGenerator(openai_api_key) #, "Single", "Married")
# for majority_sample in seed_rows:
# 	print("Majority Sample")
# 	print(majority_sample)
# 	print()
# 	print("Minority Sample")
# 	psg.generate_synthetic(majority_sample)
# 	exit()
dppsg = DPPromptSyntheticGenerator(openai_api_key, df)
dppsg.generate_new_point()