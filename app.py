import streamlit as st
from code import BiasMeasure, FindSeed, PromptSyntheticGenerator, DPPromptSyntheticGenerator
import pandas as pd
import matplotlib.pyplot as plt

# Function to read the API key from a hidden file
@st.cache(allow_output_mutation=True)
def get_api_key():
    with open(".secrets", "r") as f:
        openai_api_key = f.read().strip()
    return openai_api_key

openai_api_key = get_api_key()

# Streamlit interface
st.title("Bias-Busters DP-Gen")

with st.sidebar:
    st.title("Bias Measurement and Counterfactual Generation")
    st.write("Balancing Data with Privacy and Fairness")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    st.write("CSV File Uploaded and Randomized")
    
    # Column selection for analysis
    column_name = st.selectbox("Select the column for bias measurement:", df.columns)
    
    category1 = st.text_input("Enter the first category:", "Single")
    category2 = st.text_input("Enter the second category:", "Married")
    
    if st.button("Measure Bias"):
        bm = BiasMeasure(openai_api_key)
        regex_queries = bm.make_query(df)
        st.write("RegEx Queries Constructed")
        
        scores_df, measure_df = bm.evaluate_df(df, regex_queries)
        mean_values = measure_df.loc['mean', :].to_dict()
        category_means = {
            category1: mean_values['male'],
            category2: mean_values['female']
        }

        # Pie chart visualization
        fig, ax = plt.subplots()
        ax.pie(category_means.values(), labels=category_means.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.write("Bias Distribution in Data")
        st.pyplot(fig)
        
        fs = FindSeed(column_name)
        seed_rows, example_counterfactuals = fs.find_seeds(scores_df)
        st.write("Examples existing in the dataset:")
        st.dataframe(example_counterfactuals)
        
        psg = PromptSyntheticGenerator(openai_api_key)
        for majority_sample in seed_rows[:3]:
            st.markdown("### Majority Sample:")
            st.info(majority_sample)
            st.markdown("### Minority Sample Generated:")
            minority_sample = psg.generate_synthetic(majority_sample)
            st.success(minority_sample)
        
        # st.write("DP Synthetic Data Generation")
        # dppsg = DPPromptSyntheticGenerator(openai_api_key, df)
        # new_point = dppsg.generate_new_point()
        # st.write("New Data Point Generated:", new_point)

st.write("Ready.")