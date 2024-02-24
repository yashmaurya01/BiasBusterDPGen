from code import BiasMeasure
import pandas as pd

with open(".secrets", "r") as f:
	openai_api_key = f.read()

df = pd.read_csv("./data/biased/nlp/wino.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
bm = BiasMeasure(openai_api_key)#, "Maritial Status", "Single", "Married")
regex_queries = bm.make_query(df)
bm.evaluate_df(df, regex_queries)