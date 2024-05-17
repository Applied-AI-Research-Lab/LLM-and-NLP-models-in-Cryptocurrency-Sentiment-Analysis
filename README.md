# LLM and NLP models in Cryptocurrency Sentiment Analysis: A Comparative Classification Study

## Under Review Paper
* **Journal**: [Big Data and Cognitive Computing | An Open Access Journal from MDPI](https://www.mdpi.com/journal/BDCC)
* **Title**: LLM and NLP models in Cryptocurrency Sentiment Analysis: A Comparative Classification Study
* **DOI**: Pending

## Authors
* **Konstantinos I. Roumeliotis**
* **Prof. Nikolaos D. Tselikas**
* **Prof. Dimitrios K. Nasiopoulos**

## Abstract
Pending

## Keywords
Pending

## Methodology
This research investigates the effectiveness of large language models (LLMs), particularly GPT-4, for sentiment analysis in the cryptocurrency sector and compares its performance with other NLP models like BERT and FinBERT. To achieve this, we used the [Crypto+](https://www.kaggle.com/datasets/oliviervha/crypto-news) dataset from Kaggle, specifically curated for sentiment analysis tasks in cryptocurrency.
### Data Preparation
We employed the DatasetMethods class to prepare the dataset, splitting it into training, validation, and test sets. These sets were saved in CSV format and then processed by the GPTMethods class to create JSON files necessary for fine-tuning the GPT-4 model.
### Initial Predictions
The GPT-4 base model was first used to predict the sentiment of cryptocurrency news articles, classifying them as positive, negative, or neutral. This was facilitated by the GPTMethods class, ensuring a consistent approach.
### Model Fine-Tuning
* We fine-tuned the GPT-4 model by specifying hyperparameters and uploading the training and validation JSON sets through OpenAI's user interface. The fine-tuning process was executed, and its evaluation captured for further analysis.
* A similar methodology was applied to train and evaluate two other NLP models: BERT and FinBERT, the latter being specialized in finance. For these models, training and predictions were performed on the Google Colab platform. The corresponding Jupyter notebooks (ipynb files) used are accessible in the Colab folder.
### Sentiment Predictions
The fine-tuned GPT-4, BERT, and FinBERT models were then tasked with making sentiment predictions on the same test set used for the base GPT-4 model. Their predictions were saved in a consolidated test_set file for uniform evaluation.
### Performance Evaluation
Finally, the EvaluationMethods class was utilized to assess the performance of these models using metrics such as Mean Absolute Error (MAE), Accuracy, Precision, and F1 score. This facilitated a direct comparison among GPT-4, BERT, and FinBERT, highlighting the strengths and weaknesses of each model in the context of sentiment analysis in the cryptocurrency sector.
