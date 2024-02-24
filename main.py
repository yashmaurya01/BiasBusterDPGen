from code import BiasMeasure
import pandas as pd

with open(".secrets", "r") as f:
	openai_api_key = f.read()

df = pd.read_csv("./data/biased/tabular/adult.csv")
bm = BiasMeasure(openai_api_key)
regex_queries = bm.make_query(df)
bm.evaluate_df(df, regex_queries)