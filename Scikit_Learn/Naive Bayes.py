# Naive Bayes
# Problem
''' A company is looking to develop a text classification model to categorize customer reviews
   into positive or negative sentiments. They want to use Naive Bayes classification to automatically
   analyze and classify the reviews they receive, aiming to understand customer satisfaction levels
   and sentiments. The company desires a model that can accurately predict whether a customer review
   expresses a positive and negative sentiments '''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
''' The multinomial Naive Bayes classifier is suitable for classification
 with discrete features (e.g., word counts for text classification) '''


reviews = ["The product is excellent and work perfectly",
           "The product is very bad and disappointing",
           "Terrible product and waste of money",
           "I love this product and work amazing"]
sentiments = np.array([1,0,0,1])

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)
# fit_transform  is used to transform the reviews into a matrix of token counts (vectors)

classifier = MultinomialNB()
classifier.fit(x,sentiments)

def classify_new_reviews(new_review):
    review_vectorize = vectorizer.transform([new_review])
    predict = classifier.predict(review_vectorize)
    if predict[0] == 1:
        return "Positive"
    else:
        return "Negative"

while True:
    user_input = input("enter a review : ")
    result = classify_new_reviews(user_input)
    print(f'review is {result}')