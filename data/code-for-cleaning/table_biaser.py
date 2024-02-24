import pandas as pd

def create_subset(df, column):
    # Get counts of unique values and select the top two
    unique_values_counts = df[column].value_counts()
    
    # Select the top two unique values by count
    top_two_values = unique_values_counts.index[:2]

    # Filter the DataFrame to only include rows with the top two unique values
    filtered_df = df[df[column].isin(top_two_values)]

    # Recalculate unique values after filtering
    unique_values = filtered_df[column].unique()
    
    if len(unique_values) != 2:
        raise ValueError("Unexpected error: Filtered DataFrame does not have exactly two unique values.")

    # Sort the unique values
    unique_values_sorted = sorted(unique_values)

    # Calculate the number of samples for each attribute
    total_samples = 3000
    first_samples = int(total_samples * 0.25)
    second_samples = total_samples - first_samples

    # Create the subset
    subset_first = filtered_df[filtered_df[column] == unique_values_sorted[0]].sample(n=first_samples, random_state=1)
    subset_second = filtered_df[filtered_df[column] == unique_values_sorted[1]].sample(n=second_samples, random_state=1)

    # Combine the subsets
    subset = pd.concat([subset_first, subset_second])

    return subset

df_adult = pd.read_csv("adult.csv")
df_adult_biased = create_subset(df_adult, "sex")
df_bank = pd.read_csv("bank-full.csv", sep=';')
df_bank_biased = create_subset(df_bank, "marital")
df_compas = pd.read_csv("compas-scores-raw.csv")
df_compas_biased = create_subset(df_compas, 'Sex_Code_Text')
df_compas_biased.to_csv("compas.csv", index=False)
df_bank_biased.to_csv("bank.csv", index=False)
df_adult_biased.to_csv("adult.csv", index=False)