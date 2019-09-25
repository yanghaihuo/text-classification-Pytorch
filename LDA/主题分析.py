# 主题分析
from sklearn.datasets import fetch_20newsgroups
import random
import nltk
# nltk.download()


training_data = fetch_20newsgroups(subset='train', shuffle=True)
data500 = random.sample(training_data.data, 500)
# Clean the data
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
training_data_modified = list()
for training_example in data500:
    stemmed_words = list()

    # Find meaningful tokesn
    tokens = nltk.word_tokenize(training_example)

    # Convert words using stemming
    prt = nltk.PorterStemmer()

    for token in tokens:
        if token not in stop_words:
            if (token.isalnum() == True):
                stemmed_words.append(prt.stem(token))

    sent = ' '.join(stemmed_words)
    training_data_modified.append(sent)
training_data_cleaned = [doc.split() for doc in training_data_modified]

import gensim
from gensim import corpora
dictionary = corpora.Dictionary(training_data_cleaned)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in training_data_cleaned]
# Create model using gensim
Lda = gensim.models.ldamodel.LdaModel

# Training LDA Model
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
#Each line represent topic with words.
print(ldamodel.print_topics(num_topics=3, num_words=3))












