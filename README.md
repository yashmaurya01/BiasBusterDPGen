<img src="https://github.com/yashmaurya01/BiasBusterDPGen/blob/main/logo.png" width="30%" height="30%" align="right" />

# BiasBusterDPGen (B<sub>2</sub>DPG)

### Detailed Project Proposal: Key Steps and Sub-Steps

1. **User Data Upload Interface**: Provide a platform for users to securely upload tabular or NLP data.
   
2. **Bias Type Specification**: Allow users to specify the kind of binary bias they are concerned about in their data.

3. **Initial Data Observation**: Use LLMs to observe a subset of the data to understand its structure and content.

4. **Regex Query Generation by LLMs**: Leverage LLMs to craft two regex queries aimed at detecting the specified binary biases.

5. **Bias Detection Using Regex**: Apply the generated regex queries to the dataset to identify instances of bias.

6. **Bias Visualization**: Create pie chart visualizations to represent the proportion of detected biases within the dataset.

7. **Dataset Segmentation**: Split the dataset into three distinct subsets:
   - Majority class bias
   - Minority class bias
   - Neutral or undetermined bias

8. **Sentence Embedding with Sentence-BERT**: Embed samples from the majority and minority class bias subsets using Sentence-BERT.

9. **Cosine Similarity Analysis**: Compute cosine similarity between embeddings of majority and minority class samples to identify significant bias discrepancies.

10. **Identification of Seeds for Synthetic Counterfactuals**: Determine seeds by selecting samples from the majority class that are most dissimilar to the minority class based on cosine similarity.

11. **Synthetic Counterfactual Generation with LLMs**: Use seeds to guide LLMs in generating counterfactuals to transition samples from majority to minority class, aiming to balance the dataset.

12. **Dataset Augmentation with Synthetic Counterfactuals**: Integrate the synthetic counterfactuals into the original dataset to mitigate identified biases.

13. **Differential Privacy Synthetic Data Generator Development**:
    - **Private Dataset Consideration**: Treat the uploaded dataset as private, applying differential privacy principles.
    - **Epsilon Setting**: Allow users to set an epsilon value for differential privacy guarantees.
    - **Random Subset Sampling**: Sample a random subset from the private dataset as a basis for synthetic data generation.
    - **LLM-Powered Synthetic Data Generation**:
      - Employ in-context learning or few-shot examples to guide LLMs.
      - Generate new synthetic data by prompting LLMs, incorporating Gaussian noise into the next token prediction task for differential privacy.

### Impact and Utility

This project introduces a comprehensive framework for detecting, analyzing, and mitigating bias in datasets using state-of-the-art LLMs and NLP techniques. It not only aids in uncovering subtle biases within data but also provides a novel approach to creating balanced datasets through synthetic counterfactuals, enhancing the fairness of machine learning models derived from such data. Additionally, by incorporating differential privacy into synthetic data generation, the project addresses critical concerns regarding data privacy, making it a pioneering effort towards responsible AI development. This endeavor promises to set new standards in ethical data science practices, significantly benefiting researchers, data scientists, and organizations striving for equity and privacy in their analytical and predictive models.
