[Back to Home](index.md)

# Literature

# Chris’s Literature Review  

## [A Comparative Study of Sentiment Analysis on Customer Reviews Using Machine Learning and Deep Learning](https://www.mdpi.com/2073-431X/13/12/340)

**Goal of the Paper**  
To compare different machine learning and deep learning models for sentiment analysis of Amazon product reviews, evaluating the relative strengths and weaknesses of each. Models include neural networks (RNN, CNN) and traditional machine learning techniques (Logistic Regression, Random Forest, Naïve Bayes).

**Why is it important?**  
Helps determine the most reliable approach for interpreting complex human language in textual data. Effective sentiment analysis is crucial as natural communication grows in scale and value online, but it is challenging for machines to interpret without appropriate tools and techniques.

**How is it solved? – Methods**  
The dataset contained over 30,000 customer reviews, preprocessed as described in the paper. Models were divided into:

- **Neural Networks**  
  - Used token dictionaries (similar to embeddings) to convert words into numerical values, preserving sequential context.  
  - Models implemented in Keras.  

- **Traditional ML Models**  
  - **Logistic Regression:** Multinomial Logistic Regression to predict star ratings, trained with L-BFGS and L2 regularization.  
  - **Naïve Bayes:** Multinomial Naïve Bayes with Laplace smoothing (alpha=1).  
  - **Random Forest:** 100 decision trees using random data subsets, Gini impurity for splits, nodes fully expanded, no max leaf limit. Grid search and cross-validation applied.  

All models trained using an 80/20 train-test split.

**Results/Limitations, if any**  
- Random Forest and Logistic Regression achieved highest accuracy (~0.99).  
- RNN and CNN performed slightly lower (0.98 and 0.93).  
- Deep learning models, despite being more compute-intensive, did not outperform the best traditional ML models.  
- Highlights that in compute-limited or large-scale applications, traditional ML can provide high accuracy more efficiently.  


## [Sentiment Analysis using Support Vector Machine and Random Forest](https://www.researchgate.net/publication/378263992_Sentiment_Analysis_using_Support_Vector_Machine_and_Random_Forest)  

**What is the goal of the paper?**  
The goal of this paper was to serve as a survey of machine learning techniques (the authors use the term ‘comprehensive’ but only evaluates Random Forest and SVM) and their use in determining expressed emotion in text (specifically social media in this case). It also covers a lot of the techniques used to make these models effective, from data preprocessing to training to feature extraction. It also covers a lot of the difficulties encountered and the relative strengths and weaknesses of the various models evaluated.  

**Why is it important?**  
Sentiment analysis (opinion mining) is an NLP task that seeks to derive useful insights and information from very noisy. But it is also very easy to get large amounts of it and the near constant inflow of social media sentiment makes it a valuable and timely commodity for those that can properly handle it and extract its information value.  

**How is it solved? – Methods**  
The authors of this paper use cross-validation scores to assess the models on their training data and the standard metrics for their performance on the testing data (F1, Recall, Accuracy, Precision).  
Term frequency and inverse document frequency (TF-IDF) were used to translate textual data into numerical data in this paper (for text vectorization).  
The paper goes on to explain how a Random Forest and SVM work as well as doing a great job in explaining TF-IDF and the metrics used to evaluate the model performance, but beyond that doesn’t go into detail or specifics on model hyperparameters or optimization techniques used. It talked briefly about kernel options available to SVMs as well as options for splitting the Random Forest (Gini impurity and information gain) but not specifically which were used in this effort.  

**Results/limitations, if any.**  
Ultimately the SVM model won out over the Random Forest in all four metrics (Accuracy, Recall, Precision, and F1) as well as doing better in every instance of cross validation. The margin was not overly large with Random Forest scoring between 78 and 79 consistently, and SVM scoring 80 in all but one (second cross validation score).  
This was attributed to the SVM models’ ability to ‘effectively handle complex feature spaces, capture intricate relationships between words and sentiments, and achieve a high degree of classification accuracy’.  
The paper did acknowledge the interpretability of the Random Forest model, which has always been a valuable feature of the technique.  

## [Application of Support Vector Machine (SVM) in the Sentiment Analysis of Twitter DataSet](https://www.mdpi.com/2076-3417/10/3/1125)  

**What is the goal of the paper?**  
Unlike the other papers I read, this one focused solely on the SVM and its use as a sentiment classifier. The authors hope to demonstrate that the Fisher Kernel, as opposed to other options of the SVM is particularly well suited to making classification predictions of text data due to the kernels function being derived from the ‘Probabilistic Latent Semantic Analysis model’.  

**Why is it important?**  
The authors feel that current methods are overly dependent on statistics of sentiment words and that the use of the Fisher Kernel in SVMs will make them more effective at making classification decisions by taking into account latent semantic information contained in the text. Issues native to natural language such as synonymy and polysemy (multiple words meaning the same thing and words meaning many things) as well as the context and complex structure of language make the prospect of gleaning useful information from such raw data very difficult.  

**How is it solved? - Methods**  
The paper goes deeply into what the Fisher Kernel is and mathematically how it does it, but the interesting part is that it is then improved upon by the authors by combining the Fisher Kernel with Probabilistic Latent Semantic Analysis (PLSA). This speaks directly to the problem mentioned where currently we look at the frequency of a word like ‘awesome’ and assume it is a good thing based on definition, but don’t take into account the human capacity for sarcasm. The coupling of the PLSA with the Fisher Kernel aims to not simply take the frequency of individual words but look at the wider body of the text to determine relationships between those words to determine if they are good or bad. This makes for a much more robust solution in terms of determining sentiment (positive or negative) using an SVM.  

**Results/limitations, if any.**  
The authors used their hybridized Fisher Kernel and compared it to the results produced by other popular kernels so the SVMs compared are:  
1. HIST-SVM – Which uses simple word frequencies as features  
2. PLSA-SVM – Which just uses the PLSA as discussed but as a stand alone (no integration with the Fisher Kernel)  
3. Fisher Kernel – The improved implementation using the PLSA and the FK together  

Precision and Recall were used as metrics to compare the models.  

Ultimately the FK/PLSA SVM performed better than both of the other methods by a strong margin with the FK-SVM achieving an 87.2% precision and the HIST-SVM and PLSA-SVM achieving an 82.49% and 83.2% respectively.  
The FK-SVM also outperformed the to other two models on recall with the FK-SVM achieving an 88.3% and the HIST-SVM and PLSA-SVM achieving 83.24% and 85.67% respectively.  

## [Using VADER sentiment and SVM for predicting customer response sentiment](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1458382&utm_source=chatgpt.com&dswid=4331)  

**What is the goal of the paper?**  
The goal of this paper is to show the use of SVMs to pre-emptively deduce the sentiment of corporate emails received at a customer support service. The paper claims to show that they can even use SVMs to determine in advance how corporate email responses to customers will be received.  

**Why is it important?**  
If successful, this would allow for emails to be categorized into groups such as emails with complaints and emails from content customers. This would allow the company to prioritize responses from the company to disgruntled customers as opposed to those going back to content customers. This assumes that content customers, being content, are not as time sensitive to the replies from the company as the unhappy customers.  

**How is it solved? - Methods**  
The authors collected a dataset of over 160,000 emails from a Swedish telecom company. These emails were separated into email threads which brought the number of records down to just under 70,000. They then used Variance Aware Dictionary and sEntiment Reasoner (VADER) to assign each email a positive, negative or neutral value to be used to train their model.  

Interestingly, the Swedish telecom that the emails were collected from use their own system to sort the emails by subject. The authors discarded emails for all but the four most frequent subjects, save ‘DoNotUnderstand’ which was the most prevalent. This was done to avoid confusion in the model.  

The data was preprocessed, being sure to remove any identifying information like phone numbers, names, email headers, attachments and so on. VADER was used to assign each email a sentiment label using the Swedish lexicon.  

Each email in a thread of three or more emails is classified using VADER sequentially. This results in what the authors call a ‘compound score’ which is the ratio of positive and negative sentiment, which is then used as the label for each email in that thread. These are then used to bin the data into five bins. Very Negative (-1 to -0.65), Negative (-0.65 to -0.35), Neutral (-0.35 to 0.35), Positive (0.35 to 0.65), and Very Positive (0.65 to 1).  

Term-Frequency – Inverse Document Frequency (TF-IDF) is used to break up the emails to weight the frequency of a given term by the inverse frequency of documents that contain the term.  

Feature selection was done using Chi-Square test for the five bins with a maximum of 5,000 words being kept for each bin.  

The authors used two different SVM model (SVC and LinearSVC). These models were evaluated using F1, Accuracy, and Area Under Roc-Curve (AUC).  

Two experiments were conducted. The first experiment was to see if the algorithms are able to accurately classify the emails received into one of the five classes noted. The second experiment was to see if the same could predict the reception of response email from the company to the customer.  

Each of these two experiments had four sub-experiments, one for each of the four email topics identified earlier (Invoice, Porting, SimCardCancellation, and Assignment).  

A dummy classifier was used so that the authors could evaluate the statistical significance of the models’ results against random guessing using two non-parametric statistical tests.  

**Results/limitations, if any.**  
As the authors ran two different experiments (sentiment polarity of a given email, and the prediction of the sentiment polarity of the response email’s reception) there are two sets of results.  

1. There wasn’t a lot of difference between the two SVMs’ ability to predict which class the email belonged to with both generally achieving an F1 score above 0.8. But this was dramatically better than the random classifier, which only achieved 0.2 as expected since there were five classes and each email was randomly assigned to one of them. This showed that the model was very capable of identifying the appropriate class of a given email.  
2. Similarly, the ability of the two SVMs to predict the sentiment of a response email that hasn’t been seen yet were also very similar, but much better than the random classifier. With the F1 score of the LinearSVM being 0.667 or better and the random classifier achieving the expected 0.2. This indicates that this is an effective model for predicting the sentiment of response emails, though not quite as good as being able to assess the sentiment of emails that have been received.  

## [Sentiment Analysis of 2024 Presidential Candidates Election Using SVM Algorithm](https://trilogi.ac.id/journal/ks/index.php/JISA/article/view/1714/878)

**What is the goal of the paper?**  
The goal of this paper is to attempt to use sentiment analysis of social media (specifically Twitter (Primary), LinkedIn, Facebook, and YouTube are mentioned) to predict election outcomes for presidential elections. Based on the information in the document regarding the publisher and University, it appears this is regarding the presidential election of Indonesia, but the goal could just as easily be applied to any election that produced a sufficient body of data on social media of course.

**Why is it important?**  
Making sense of social media and deriving useful information from it is a very useful and even profitable effort. In this particular paper we see that the focus is on political campaigning which would be of incredible value to not just the voting body, but to the candidates, their supporters and their campaign as they would ideally be able to judge public reception of their platforms and allow for near real time adjustment of their campaign strategy.

**How is it solved? – Methods**  
The authors followed a pretty standard methodology of scraping Twitter for tweets that mentioned the candidate name and ‘President’ for a period that covered the first quarter of 2023 and then removing duplicates.  

Next, they preprocessed the data using common methods like lowercasing, stemming, removing stop words, tokenizing, translating slang terms, and removing special characters like punctuation, numbers and emojis. I think they missed a couple of opportunities to improve their process. Namely keeping emojis and the slang terms and simply assigning them values (positive, negative or neutral). The authors also removed synonyms that appear in the same sentence which I thought was less than ideal.  

Using TF-IDF (Term-Frequency – Inverse Document Frequency) the assessed the value of each word that remained. This is a very common and powerful technique that a lot of researchers have had success with, and I’m not surprised to see it here.  

After training the SVM using a 9:1 training testing split, they used 10-fold cross validation to assess the performance of the model using the F1 score.  

Finally, they tested it and presented their results.  

**Results/limitations, if any.**  
The results of the work concluded with the model with the best fold based on F1 score and an analysis for the ‘linkages of the predicted model results using the Pearson Moment Product Correlation’. The authors found that there was indeed a relationship between ‘electability survey and the results of the analysis of positive sentiment’.

## [Travel VLOG Reviews: Support Vector Machine Performance in Sentiment Classification](https://www.iieta.org/journals/isi/paper/10.18280/isi.300109)

**What is the goal of the paper?**  
This paper is looking to demonstrate improvement of SVMs to conduct sentiment analysis on unbalanced datasets using Synthetic Minority Over-sampling TEchnique (SMOTE). Specifically, it is aimed at producing a better methodology to understand tourist behavior in the travel industry but if successful, such a technique could probably be applied to any appropriate classification problem with an unbalanced dataset.  

**Why is it important?**  
The continued advancement of online platforms and the integration of consumers with the wider community gives companies a real imperative to be responsive to changing customer demands and allows them to remain agile in keeping promotions, marketing efforts and customer satisfaction aligned in near real time. It also allows them to stay aware of trends and gain insights that ultimately keep competition high, providing for better product offerings to consumers.  

User generated content with relation to the tourism industry is particularly suited to such analysis due to the unfiltered and constant flow of opinions and experiences by VLOGers to their community of followers and others doing pre-travel research.  

**How is it solved? – Methods**  
Data is first pre-processed of course, and the authors used Term Frequency – Document Inverse Frequency to vectorize the text data with `max_features=5000` and `ngram_range=(1,2)` for parameters. MinMax scaling was used to scale the data and Recursive Feature Elimination (RFE) was used for feature selection, using an SVM to select the top 20% interestingly. Principle Component Analysis (PCA) was also used to further slim down the features/dimensionality. Finally, the data was binned using equal width binning. Sentiment scores were discretized into Negative, Neutral, and Positive.  

SVM was augmented by using SMOTE to boost the under-represented minority classes. The parameters used for SMOTE in this instance were `sampling_strategy=0.5`, `k_neighbors=5`, and `random_state=42`.  
The SVM was run both with and without SMOTE to demonstrate the ability for SMOTE to enhance the results generated by the SVM.  

**Results/limitations, if any.**  
Without using SMOTE the SVM model was still able to achieve a high accuracy score of 95%, but the authors report that the confusion matrix shows it was almost completely unable to properly classify the minority class with just 4 true negatives and 57 false negatives. The optimistic AUC was 0.886 while the pessimistic AUC was a mere 0.239.  

Using SMOTE with the SVM achieved worse accuracy coming in at 83.12% but a better result in accurately classifying the under-represented class. The optimistic AUC was 0.978 and the pessimistic AUC was 0.969, demonstrating a much more general ability to predict classes under different scenarios.

---

# Hunter's Literature Review

## [Twitter sentiment analysis and bitcoin price forecasting: implications for financial risk management](https://www.proquest.com/docview/3047039752?accountid=14787&pq-origsite=primo&sourcetype=Scholarly%20Journals)

**What is the goal of the paper?**  
The goal of this paper is to analyze sentiment analysis using gathered Twitter data to predict movements in price of Bitcoin, a digital currency.  

**Why is it important?**  
Bitcoin was introduced with an initial price of $0.00099 in 2009 and currently has a valuation of $114,332 per coin which makes the possibility of predicting its price movements a highly valuable area of research.  

**How is it solved? – Methods**  
The data used was collected by scraping Twitter data using the `snscrape` tool then preprocessing by eliminating hashtags, mentions, and https links. Any tweets left empty were then excluded. To create the test dataset, 10 independent human raters were hired to analyze each entry and assign a sentiment score.  

As opposed to similar studies, neutral tweets were left in the final dataset as they were determined to be statistically significant in the final model. The researchers utilized VADER for sentiment analysis. It was determined that trying to predict precise price movements was cumbersome and unnecessary, so they utilized logistic regression to predict whether the price would rise or fall.  

**Results/limitations, if any.**  
The results were unfortunately lackluster with accuracy, precision, recall, and F-1 scores hovering in the mid to high 50% range. The model did reveal some interesting discoveries, notably that negative sentiment has a significantly higher impact on price as opposed to positive sentiment.  

One of the main limitations of the study was using only English tweets as it excluded the sentiments of all non-English speaking Twitter users. Overall this article was very informative and will be heavily utilized as an example for our research.  

---

## [A scoping review of preprocessing methods for unstructured text data to assess data quality](https://pmc.ncbi.nlm.nih.gov/articles/PMC10476151/)

**What is the goal of the paper?**  
The goal of this paper is to review a variety of available methods for pre-processing unstructured text data (UTD) and assess their performance as it pertains to data quality after pre-processing.  

**Why is it important?**  
This area of research is important due to the large amounts of UTD data available in medical records, social media, and a variety of additional sources, but quality pre-processing methods are crucial for turning this data into useful insights.  

**How is it solved? – Methods**  
The method employed for this research was to find peer-reviewed journal articles from 2002–2021, exclude all but those most pertinent to finding the gold standard for UTD preprocessing and preparation, and perform a detailed analysis to find the most effective methods available.  

Of all the studies included, the most often utilized aspect of data pre-processing was restructuring and reorganization techniques including removing stop words, removing punctuation, removing URLs, tokenization, converting to lowercase, and many more.  

**Results/limitations, if any.**  
Two of the main limitations of the study were only reviewing articles in English and grey literature not being searched.  

---

## [The applications of artificial neural networks, support vector machines, and long–short term memory for stock market prediction](https://www.sciencedirect.com/science/article/pii/S2772662221000102)

**What is the goal of the paper?**  
This article focuses on an in-depth analysis of stock market prediction using artificial neural networks, support vector machines, and long-short term memory networks. The article begins by providing a high-level overview of the stock market, machine and deep learning models, and the potential for prediction.  

**Why is it important?**  
This research is important as publicly traded stocks is a multi-trillion dollar industry and an advantage in predicting price movement would be extremely lucrative for companies as well as private investors.  

**How is it solved? – Methods**  
The authors divided the report into three sections: ANNs, SVMs, and LSTM networks. Each section reviewed multiple attempts at predicting stock market movements using the aforementioned models. The studies did not reveal one model that consistently performed the best, but it seems that each model has negatives and positives depending on accuracy, performance, and amount of data required.  

Finally, the authors conclude that there is high potential for stock market prediction and investment advice using machine learning, deep learning, and AI.  

**Results/limitations, if any.**  
The authors surmise that investment and advancement in these technologies will only continue to increase and become more financially and technologically feasible for stock market prediction.  

## [Exploring Kernel Machines and Support Vector Machines: Principles, Techniques, and Future Directions](https://www.mdpi.com/2227-7390/12/24/3935)

**What is the goal of the paper?**  
This article explores an introductory but in depth review of Kernel and Support Vector Machines. The first half of the article begins with a deep dive of information into the history of Kernel machines before moving to discussing the mathematical theories and constructs behind the function of Kernel Machines. Finally, the authors discuss the math, benefits, and drawbacks of various types of Kernel Machines available and some of the problems each may help to solve.  

The latter half of the article follows a similar format in discussing the history, mathematical theories, and applications of Support Vector Machines. Many types of SVMs were discussed in depth including Standard, Multi-Class, One-Class, Incremental, and SVMs for active, transductive, and semi-supervised learning. The varieties of each type were discussed including the underlying equations, pros, and cons of each. Finally, the article discusses the future potential of integrating SVMs into deep learning architectures and the potential for increased efficiency and scale that could be acquired.  

**Why is it important?**  
Kernel and Support Vector Machines are powerful machine learning models utilized in a multitude of industries such as healthcare, finance, technology, and many more.  

---

## [Predicting Stock Movement Using Sentiment Analysis of Twitter Feed with Neural Networks](https://www.scirp.org/pdf/jdaip_2020111613521357.pdf)

**What is the goal of the paper?**  
The goal of this paper is extremely similar to the objective of our project; perform sentiment analysis on a large collection of tweets using a machine learning or deep learning model and then use those sentiments to predict movement in the stock price of publicly traded companies.  

**Why is it important?**  
This research is important as publicly traded stocks is a multi-trillion dollar industry and an advantage in predicting price movement would be extremely lucrative for companies as well as private investors.  

**How is it solved? – Methods**  
First, the team began by performing sentiment analysis using Support Vector Machines (SVM) as it was the highest performing model with 0.83 accuracy. The dataset used was the Sentiment 140 dataset from Kaggle composed of 1.6 million tweets collected using the Twitter API.  

Next, a Multilayer Perceptron Neural Networks (MLP) model was chosen due to its higher accuracy over a Boosted Regression Trees model and was used to predict the closing prices of various stocks and the Dow Jones Industrial Average. No limitations were listed for this research.  

**Results/limitations, if any.**  
The results were surprisingly accurate with analysis focusing on AAPL stock using SVM for sentiment analysis and MLP performing the best with only a 0.98 Mean Absolute Error. This could signify that focusing on a specific company as opposed to the overall stock market or Dow Jones Industrial Average could be the best method to ensure more accurate results.  

---

## [Predicting stock movement using sentiment analysis of Twitter feed](https://ieeexplore.ieee.org/document/8338584)

**What is the goal of the paper?**  
I chose this article as it was heavily cited and seemed to be a direct motivation for the authors of the previous article I read. The goal of this paper is to analyze the sentiment of Twitter data then use a Boosted Regression Tree classifier for predicting stock price movement for the day following the tweet. The dataset chosen was the Sentiment 140 dataset on Kaggle composed of 1,578,614 tweets collected through Sentiment 140 API.  

**Why is it important?**  
This research is vital as publicly traded stocks is a multi-trillion dollar industry and an advantage in predicting price movement would be extremely lucrative for private investors and companies.  

**How is it solved? – Methods**  
Pre-processing of the Twitter data followed five steps: all characters changed to lower case, all URLs removed, reduced all tweets to a single whitespace, usernames removed, removal of retweet signs and quotes. The dataset was then split into 80/20 training and test sets.  

A Support Vector Machines model was used for sentiment analysis of the processed data, then a Boosted Regression Trees model was used for prediction of stock price.  

**Results/limitations, if any.**  
The resulting predictions were extremely accurate during timelines with mild stock market price changes, but the model was less accurate during high stock price volatility. While this study had very promising results, the authors did note that they would make some changes during future research including using a Recurrent Neural Network for stock prediction.  

The main limitation listed is that the authors believe that training on a timeline of greater than one year could improve predictive performance in future work.  

# Camille's Literature Review

## [Financial Sentiment Analysis: Techniques and Applications](https://dl.acm.org/doi/pdf/10.1145/3649451)

**What is the goal of the paper?**  
The paper is a survey whose goal is to provide a comprehensive review of financial sentiment analysis (FSA), covering both the techniques used for FSA and its applications in financial markets. It also aims to clarify the scope of FSA, define its relationship to investor sentiment and market sentiment, and propose frameworks to understand how technique and application research intersect.  

**Why is it important?**  
Sentiment analysis in finance presents unique challenges, including specialized jargon, numerical and textual hybrid data, and domain-specific interpretations. Understanding investor sentiment from sources like news, social media, and company filings can yield complementary signals for forecasting stock movements, managing risk, and supporting financial decision-making. This survey connects techniques and applications, providing a holistic view of FSA’s role.  

**How is it solved? – Methods**  
The authors conducted a structured literature review, distinguishing between technique-driven research (datasets, models, algorithms) and application-driven research (how sentiment improves financial predictions). They review lexicon-based techniques, traditional machine learning, hybrid models, deep learning, and transformer-based models, covering different sentiment granularities (sentence-level, aspect-level, intensity scoring). They also examine datasets, feature extraction, embeddings, evaluation metrics, and applications in stock prices, risk, portfolio optimization, forex, and cryptocurrency. Trends, gaps, and open research challenges are synthesized.  

**Results/limitations, if any.**  
FSA has evolved toward deep learning and transformer models with implicit sentiment embeddings improving predictive performance. Negative sentiment often has a stronger and longer-lasting effect than positive sentiment. Combining multiple sentiment sources yields more robust forecasts. Limitations include poor generalizability across sectors/time, high cost/scarcity of annotated data, lack of interpretability, noisy signals, and difficulty quantifying nuanced sentiment like sarcasm or metaphor.  

---

## [Sentiment Analysis Stock Market: Sources and Challenges](https://research.aimultiple.com/sentiment-analysis-stock-market)

**What is the goal of the paper?**  
The goal is to explain what stock market sentiment analysis is, why it matters, and how it can be used to better predict stock price movements. It provides an overview of data sources, methods, accuracy levels, and challenges for applying sentiment analysis in finance.  

**Why is it important?**  
Stock market sentiment captures investor psychology, influencing stock price fluctuations alongside traditional indicators. Sentiment from news, social media, and financial reports allows investors to anticipate market shifts and refine strategies. Research shows incorporating sentiment can improve prediction accuracy by up to 20%.  

**How is it solved? – Methods**  
The paper describes NLP and machine learning applied to data from news feeds, websites, social media (Twitter, Reddit), financial reports, and economic indicators. Steps include data collection, preprocessing (tokenization, noise removal), and labeling as positive, negative, or neutral. Methods include rule-based systems, lexicon-based approaches, and advanced models like XGBoost and BERT, which uses context-aware embeddings and attention to extract sentiment.  

**Results/limitations, if any.**  
Accuracy ranges from 60%–99% depending on dataset and model. Combining sentiment with technical/fundamental indicators improves reliability. Large language models like GPT have shown strong portfolio returns. Challenges include noisy data, evolving language, integrating qualitative with quantitative metrics, and regulatory compliance. Sentiment analysis is a complementary tool requiring continual fine-tuning.  

---

## [Stock Market Forecasting Based on Text Mining Technology: A Support Vector Machine Method](https://arxiv.org/abs/1909.12789)

**What is the goal of the paper?**  
To investigate whether text mining combined with SVM models can improve prediction of Chinese stock market trends and prices by integrating online news data with market data.  

**Why is it important?**  
Predicting stock markets is difficult due to volatility and multiple influencing factors. In China, news can heavily affect retail investors. The paper demonstrates systematic application of text mining to enhance price prediction.  

**How is it solved? – Methods**  
Collected 2.3M Chinese financial news articles (2008–2015) and stock data for 20 stocks. Developed sentiment dictionaries scoring words –5 to +5. Constructed daily input vectors combining sentiment features with market variables. Trained SVR for price prediction and SVC for trend prediction using LIBSVM with polynomial/sigmoid kernels. Parameters optimized via grid search and genetic algorithms; experiments tested time lags and data expansion.  

**Results/limitations, if any.**  
SVR achieved SCC ≈ 98.5% and MSE ≈ 0.00328, capturing sharp price fluctuations with a 1–2 day delay. SVC achieved ~59% classification accuracy. News impacts last <2 days; γ parameter is critical. Limitations include low news volume reducing accuracy, partial dictionaries, and need for more text sources for generalization.  

---

## [Using News to Predict Investor Sentiment: Based on SVM Model](https://www.sciencedirect.com/science/article/pii/S187705092031588X?via%3Dihub)

**What is the goal of the paper?**  
To explore how news influences investor sentiment and build an SVM model to predict changes in sentiment, using the psychology line (PSY) as an indicator.  

**Why is it important?**  
Previous studies focused on news affecting stock prices directly, ignoring investor emotions. Predicting sentiment can guide rational decision-making and improve cumulative returns.  

**How is it solved? – Methods**  
Collected 19,000 company news texts and 10,000 industry news texts (2013–2017, medical sector). Cleaned and segmented using Python jieba, vectorized via doc2vector, and labeled PSY >/< 50. Trained SVM to classify positive/negative sentiment, tested U-SVM for industry news, and compared data sizes/indicators.  

**Results/limitations, if any.**  
SVM achieved ~59% accuracy using PSY. Firm-specific news outperformed industry news; accuracy declined with less data. Limitations: modest accuracy, single-sector focus, overfitting risk. Future work: multi-sector and hybrid models.  

---

## [Twitter Sentiment Geographical Index Dataset](https://www.nature.com/articles/s41597-023-02572-7)

**What is the goal of the paper?**  
Introduce the Twitter Sentiment Geographical Index (tSGI), an open-source dataset of location-specific sentiment from 4.3B geotagged tweets since 2019, enabling global, multilingual sentiment analysis at daily and county/city resolution.  

**Why is it important?**  
Traditional well-being measures are costly and infrequent. tSGI offers real-time, large-scale sentiment indicators for monitoring trends, crisis responses, and cross-country comparisons, complementing economic and health metrics.  

**How is it solved? – Methods**  
Used Harvard CGA Geotweet Archive; preprocessed text (removed URLs/emojis, normalized mentions, truncated to 52 words). Sentiment computed via multilingual S-BERT embeddings across 50+ languages, classified with a 4-layer neural network trained on Sentiment140 (83% accuracy). Sentiment probabilities aggregated daily at multiple spatial levels.  

**Results/limitations, if any.**  
Covers 164 countries, ~3M geotagged tweets/day. Outperforms dictionary-based approaches; strong correlation with independent well-being indices. Limitations: sampling bias (geotagged tweets), population representativeness, and no socio-demographic data.  

---

## [A Comparative Study of Machine Learning Algorithms for Stock Price Prediction Using Insider Trading Data](https://arxiv.org/abs/2502.08728v2)

**What is the goal of the paper?**  
Evaluate multiple ML algorithms (Decision Trees, Random Forests, SVMs with various kernels, K-Means) for predicting stock prices using insider trading data, identifying the most accurate and computationally efficient methods.  

**Why is it important?**  
Insider trading signals company sentiment, potentially predicting price movements. Knowing which algorithms perform best on such data guides investors and analysts. Combines alternative data with predictive analytics for data-driven finance.  

**How is it solved? – Methods**  
Collected 1,997 Tesla insider trading records (Apr 2020–Mar 2023). Engineered features (Dollar Volume, Transaction Type), applied RFE to select Shares, Transaction Date, Dollar Volume, and Type. Dataset split 70/30 train/test. Trained Decision Tree, Random Forest, SVM (linear/poly/RBF), K-Means, evaluated by accuracy and runtime.  

**Results/limitations, if any.**  
SVM-RBF achieved highest accuracy 88%, Random Forest 83%; Decision Tree fastest but least accurate 68%, K-Means 73%. Higher computational cost correlated with better accuracy. Limitations: small dataset (Tesla-only), no macroeconomic/news data; future work could integrate more data sources.  

[Back to Home](index.md)
