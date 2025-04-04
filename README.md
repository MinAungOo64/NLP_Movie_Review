# Movie Reviews Sentiment Analysis

## About the Project

This small project focuses on **sentiment analysis** of movie reviews using the **IMDb dataset** containing 50,000 reviews. The task is to classify the movie reviews as either **positive** or **negative**.

### Dataset: IMDB 50K Movie Reviews

The dataset consists of:

- **50,000 movie reviews**.
- The task is to predict the sentiment (positive or negative) of each review using **pre-trained models**.

For more information about the dataset, visit:  
[IMDb Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

The dataset contains two columns:
- **`review`**: The text of the movie review.
- **`sentiment`**: The sentiment of the review, either "positive" or "negative".

---

## Text Processing

The following **text preprocessing** steps were applied to the dataset to clean and prepare the text for sentiment analysis:

- **Lowercasing**: All text was converted to lowercase to ensure uniformity.
- **Stop Words Removal**: Common words (e.g., "the", "and", "is") were removed to reduce noise.
- **URL Removal**: Any URLs present in the text were removed.
- **Emoji Removal**: Emojis were removed as they may introduce noise for text-based models.
- **HTML Tag Removal**: HTML tags were stripped from the text.
- **Lemmatization**: Words were lemmatized to their root form, improving the model's understanding of word variations.

---

## Sentiment Analysis with Hugging Face

For sentiment analysis, we utilized the **Hugging Face `pipeline` API** with the pre-trained **DistilBERT model** fine-tuned on the SST-2 dataset. The model was used directly for prediction without any additional training.

```python
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
```
### Model Performance

The performance of the model was evaluated under different preprocessing configurations. The accuracy was calculated based on the number of correct sentiment predictions made by the model:

| Preprocessing Configuration                       | Accuracy |
|--------------------------------------------------|----------|
| With Lemmatization and Stop Word Removal         | 0.7888   |
| Without Lemmatization and With Stop Word Removal | 0.8050   |
| Without Lemmatization and Without Stop Word Removal | 0.8611   |
