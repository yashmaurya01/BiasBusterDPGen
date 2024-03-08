from code import BiasMeasure, FindSeed, PromptSyntheticGenerator, DPPromptSyntheticGenerator
import pandas as pd

with open(".secrets", "r") as f:
	openai_api_key = f.read()

print("Read API Key")
df = pd.read_csv("./data/biased/tabular/adult.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# bm = BiasMeasure(openai_api_key)#,"Maritial Status", "Single", "Married")
# regex_queries = bm.make_query(df)
# print()
# print("RegEx Done")
# scores_df = bm.evaluate_df(df, regex_queries)
# print()
# print("Score DF Generated")
# print(scores_df)
# fs = FindSeed("Maritial Status")#,"Single", "Married")
# print()
# print("Model Loaded")
# seed_rows, example_counterfactuals = fs.find_seeds(scores_df)
# print("Examples existing in the dataset:\n", example_counterfactuals)
# print()
# print("Seed Found")
# psg = PromptSyntheticGenerator(openai_api_key) #, "Single", "Married")
# for majority_sample in seed_rows:
# 	print("Majority Sample")
# 	print(majority_sample)
# 	print()
# 	print("Minority Sample")
# 	psg.generate_synthetic(majority_sample)
# 	break
print()
print("DP Synthetic Data Generation")
dppsg = DPPromptSyntheticGenerator(openai_api_key, df)
dppsg.generate_new_point()