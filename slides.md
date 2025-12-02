# Welcome to Our Project
Use your arrow keys to navigate

---

### Using Twitter Sentiment Analysis to Predict Stock Price Movements of Major Companies via Support Vector Machines

##### Authors: Chris Charro, Hunter Clayton, Camille Esteves

---

# Introduction

---

### Sentiment Analysis

- Sentiment analysis is the process of analyzing text to derive the underlying emotions being expressed in a collection of data.
  - Allows companies to understand how stakeholders ***feel*** about their brand, products, or direction at a large scale.
  - Used across a variety of industries including technology, finance, and retail.

---

### Twitter(X)'s Role & Advantages for Sentiment Analysis

- With over 586 million active users, Twitter(X) contains a wealth of user sentiment from all over the globe.
  - Every day, millions of users flock to the site to share their opinions, news, humor, and feelings about a multitude of topics.

---

### Twitter(X)'s Role & Advantages for Sentiment Analysis (Continued)

- With a maximum character limit of 280, Twitter(X) is the premier social media site for short-form content.
  - Short form content has several benefits for sentiment analysis:
    - Less computationally intensive
    - Reduced storage requirements
    - Greater availability of training data

---

### Why is Sentiment Analysis Important

- The constant flow of data makes Twitter(X) an invaluable resource for analyzing trends in public sentiment.
- In financial markets, these sentiments can serve as indicators of investor confidence, public reactions, and potential price fluctuations.
- Prior studies demonstrate that analyzing sentiment from Twitter posts can effectively reflect investor sentiment and broader market trends

---

### Goal of our project

- *Perform sentiment analysis on Twitter(X) data related to publicly traded companies to determine the possibility and accuracy of predicting the price movements of stock prices based on user sentiment.*

---

# Methods

---

### Sentiment Analysis

- We selected three primary methods for sentiment Analysis:
  - Valence Aware Dictionary and sEntiment Reasoner (VADER)
  - Financial Valence Aware Dictionary and sEntiment Reasoner (finVADER)
  - Financial Bidirectional Encoder Representations from Transformers (finBERT)

---

### VADER

- Developed in 2014 by C.J. Hutto and Eric Gilbert at the Georgia Institute of Technology.
- Lexicon based with librry of common words and phrases. Assigns a score to each word and aggragates to determine positive, neutral, or negative sentiment.
- Designed specifically for "micro-blog like text" used in social media sites like Twitter(X).
- Requires no training data which allows extremely fast operation. 

---

### finVADER

- Developed in 2023 by Petr Koráb.
- Open-source adaptation of VADER model with additional financial domain-specific lexicons.
- Trained on financial texts like earnings reports, news articles, and finance specific tweets.

---

### BERT

- Deep learning language model developed by researchers at Google in 2018.
- Revolutionary in Natural Language Processing by introducing deep bidirectional training or transformer encoders.
- Bidirectional mechanism considers preceding and following context simoultaneously, allowing deeper language understanding.
- Excels at a variety of tasks including question answering, search query ranking, next-sentence prediction, and sentiment analysis.

---

### finBERT

- Open-source, domain-specific adaptation of BERT model introduced by Dogu Tan Araci in 2019.
- Pre-trained and fine-tuned on financial texts to deliver improved performance on data involving uncommon financial specific terminology.
- Requires far more computational resources and training times when compared to lexicon based approaches such as VADER.

---

### Support Vector Machines (SVM)

- Once sentiment analysis is complete, SVM will be used to make predictions on stock price movement.
  - SVM is a Supervised machine learning method that excels in classification tasks.
  - Commonly used in financial research for it's ability to detect patterns in market movement.
  - Prior studies, Notably by Chakraborty et al., have shown SVM models outperform other approaches in stock price prediction. 

---

### Support Vector Machines (SVM)

Supervised machine learning method that functions by using a linear decision function

![SVM Decision Function](images/svm_formula.png)

---



