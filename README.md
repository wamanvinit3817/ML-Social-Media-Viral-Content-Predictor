Social Media Virality Prediction

This project focuses on predicting whether a social media post will become viral or non-viral using supervised machine learning. The problem is modeled as a binary classification task with a significantly imbalanced target variable (is_viral), making metric selection and leakage prevention critical.
Note - All the ML algorithms used are not just implemented using sklearn libraries, they are studied mathematically and even some of them are implemented from scratch like KNN,Linear Regression etc.

Dataset & Features

Each data point represents a social media post with metadata including:

Categorical features: platform, content_type, topic, language, region

Numerical features: views, likes, comments, shares, engagement_rate, sentiment_score

Text feature: hashtags

Datetime feature: post_datetime

Target: is_viral (1 = viral, 0 = non-viral)

Preprocessing & Feature Engineering

Categorical variables were encoded using One-Hot Encoding. The post_datetime column contained date-only information, so temporal features such as post_weekday, post_month, post_day, and a binary is_weekend indicator were derived.
Hashtags, being multi-label text data, were encoded using CountVectorizer with hashtag-specific tokenization and minimum document frequency filtering to reduce noise. To avoid data leakage, the train–test split was performed before fitting the vectorizer.

Models Implemented

The following models were trained and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN) with hyperparameter tuning over multiple values of k

Decision Tree Classifier

Class imbalance was addressed using techniques such as class_weight="balanced" where applicable.

Evaluation Strategy

Due to class imbalance and the requirement to correctly classify both viral and non-viral posts, accuracy was avoided. The primary evaluation metric was F1-score, supported by precision, recall, confusion matrix, and ROC-AUC for probabilistic models. Decision thresholds were analyzed to balance false positives and false negatives.

Results & Conclusion

Logistic Regression achieved the best overall performance with the highest F1-score and strong generalization, while KNN showed high recall but slightly lower precision at larger k. Decision Trees initially overfit, highlighting the importance of controlled model complexity. Overall, the project demonstrates that proper preprocessing, leakage-safe pipelines, and correct metric selection are more important than model complexity in imbalanced classification problems.
