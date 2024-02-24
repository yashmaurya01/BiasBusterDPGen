from code import BiasMeasure, FindSeed
import pandas as pd

with open(".secrets", "r") as f:
	openai_api_key = f.read()

df = pd.read_csv("./data/biased/tabular/adult.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
bm = BiasMeasure(openai_api_key)#, "Maritial Status", "Single", "Married")
regex_queries = bm.make_query(df)
scores_df = bm.evaluate_df(df, regex_queries)
fs = FindSeed()
fs.find_seeds(scores_df)