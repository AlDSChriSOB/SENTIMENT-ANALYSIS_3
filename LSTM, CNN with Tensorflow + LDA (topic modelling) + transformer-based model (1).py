#!/usr/bin/env python
# coding: utf-8

# ### Exploratory Data Analysis and Data Processing

# In[2]:


import os
for dirname, _, filenames in os.walk("G:\SECOND SEMESTER\I CAN PLUS TO GOD BE ALL GLORY\APPLIED ARTFICIAL INTELLIGENCE\GRADED FOLDER\SARCASM DETECTION IN SENTIMENT ANALYSIS"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[6]:


import tensorflow as tf
import pandas as pd
import json
import string
import re
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import nltk
import spacy
import sys
from spacy.lang.en import English
#from spacy import en_core_web_sm             
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


#import numpy as np 
#import pandas as pd 
#import os
#import matplotlib.pyplot as plt 
import seaborn as sns 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px
from plotly.offline import init_notebook_mode
#import re
#import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
#import spacy

tqdm.pandas()
spacy_eng = spacy.load("en_core_web_sm")
nltk.download('stopwords')
lemm = WordNetLemmatizer()
init_notebook_mode(connected=True)
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (20,8)
plt.rcParams['font.size'] = 18

print(tf.__version__)  # 2.0.0-beta0


# #### Text Classification: Using Transformers Encoder Block
# A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data
# They hold the potential to understand the relationshipbetween sequential elements that are far from each other
# They are way more accurate
# They pay equal attention to all the elements in the sequence
# 

# In[7]:


data1 = pd.read_json("G:\\SECOND SEMESTER\\I CAN PLUS TO GOD BE ALL GLORY\\APPLIED ARTFICIAL INTELLIGENCE\\GRADED FOLDER\\SARCASM DETECTION IN SENTIMENT ANALYSIS\\Sarcasm_Headlines_Dataset.json\\Sarcasm_Headlines_Dataset.json", lines=True)
data2 = pd.read_json("G:\SECOND SEMESTER\I CAN PLUS TO GOD BE ALL GLORY\APPLIED ARTFICIAL INTELLIGENCE\GRADED FOLDER\SARCASM DETECTION IN SENTIMENT ANALYSIS\Sarcasm_Headlines_Dataset_v2.json\Sarcasm_Headlines_Dataset_v2.json", lines=True)


# In[8]:


data1 = data1[['headline','is_sarcastic']]
data2 = data2[['headline','is_sarcastic']]

data = pd.concat([data1,data2])
data.reset_index(drop=True, inplace=True)


# In[9]:


data


# ### EDA and Text Preprocessing

# #### Checking for Missing Values

# In[10]:


data.isnull().sum()


# #### Finding the Classes Balance/Imbalance
# 

# In[11]:


px.bar(data.groupby('is_sarcastic').count().reset_index(), x='headline',title='Count of Sarcastic and Genuine Headlines')


# #### Special Characters Removal
# 

# We will not remove numbers from the text data right away, lets further analyse if they contain any relevant information
# We can find the entity type of the tokens in the sentences using Named Entity Recognition (NER), this will help us identify the type and relevance of numbers in our text data

# In[12]:


stop_words = stopwords.words('english')
stop_words.remove('not')

def text_cleaning(x):
    
    headline = re.sub('\s+\n+', ' ', x)
    headline = re.sub('[^a-zA-Z0-9]', ' ', x)
    headline = headline.lower()
    headline = headline.split()
    
    headline = [lemm.lemmatize(word, "v") for word in headline if not word in stop_words]
    headline = ' '.join(headline)
    
    return headline


# In[13]:


def get_entities(x):
    entity = []
    text = spacy_eng(x)
    for word in text.ents:
        entity.append(word.label_)
    return ",".join(entity)

data['entity'] = data['headline'].progress_apply(get_entities)


# In[14]:


data['clean_headline'] = data['headline'].apply(text_cleaning)


# In[15]:


data['sentence_length'] = data['clean_headline'].apply(lambda x: len(x.split()))
data


# #### Headlines Length Distribution
# Look for outlier length of headline sentences
# Usually the headlines shouldn't be more than 20-30 words

# In[16]:


px.histogram(data, x="sentence_length",height=700, color='is_sarcastic', title="Headlines Length Distribution", marginal="box")


# In[17]:


data[data['sentence_length']==107]['headline']


# In[18]:


data.drop(data[data['sentence_length'] == 107].index, inplace = True)
data.reset_index(inplace=True, drop=True)


# #### Headlines Length Distribution: Outliers Removed
# The headlines after the removal of outliers do not exceed the limit of 20-30 words
# They are mostly centered in the range of 5-10 words
# 

# In[20]:


px.histogram(data, x="sentence_length",height=700, color='is_sarcastic', title="Headlines Length Distribution", marginal="box")


# #### Filtering: Find Sentences that Contain Numbers

# In[21]:


data['contains_number'] = data['clean_headline'].apply(lambda x: bool(re.search(r'\d+', x)))
data


# #### Analysis of Samples Containing numbers of Time, Date or Cardinal Entity type
# The numbers in a text data can have different implications
# While the naive text preprocessing methods suggest that the numbers should be removed along with the special characters
# The entity type of these numbers should be identified to get their exact implications

# #### 10 Random Samples: Date Entity

# In[22]:


data[(data['contains_number']) & (data['sentence_length']<=5) & (data['entity']=='DATE')].sample(10)


# #### 10 Random Samples: Time Entity

# In[23]:


data[(data['contains_number']) & (data['sentence_length']<=5) & (data['entity']=='TIME')].sample(10)


# ##### 10 Random Samples: Cardinal Entity

# In[24]:


data[(data['contains_number']) & (data['sentence_length']<=5) & (data['entity']=='CARDINAL')].sample(10)


# Inference: A lot of these headlines wouldn't make sense without these time,date or even cardinal numbers. For now we can let them be a part of our clean text data, in the next version of this notebook we will try to figure out if we can replace these numbers with specific tokens so that the meaning of them is not completely lost by removing them. Also the vocabulary size can be reduced after this step.

# ##### Word Visualization: Word Clouds

# In[25]:


sarcastic = data[data['is_sarcastic']==1]['clean_headline'].tolist()
genuine = data[data['is_sarcastic']==0]['clean_headline'].tolist()


# ### Top 50 Words: Sarcastic Headlines

# In[26]:


wordcloud = WordCloud(max_words=50, width=600, background_color='white').generate(" ".join(sarcastic))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### Top 50 Words: Genuine Headlines
# 

# In[27]:


wordcloud = WordCloud(max_words=50, width=600, background_color='white').generate(" ".join(genuine))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ### GO TO BOTTOM FOR MODEL

# ## EDA 2

# In[3]:


# Run this code for the first time, to install the libraries and download wordnet
# %reset
# !{sys.executable} -m pip install spacy
# !{sys.executable} -m spacy download en
# !{sys.executable} -m pip install pyLDAvis
# !{sys.executable} -m pip install gensim
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[4]:


df = pd.read_json("G:\SECOND SEMESTER\I CAN PLUS TO GOD BE ALL GLORY\APPLIED ARTFICIAL INTELLIGENCE\GRADED FOLDER\SARCASM DETECTION IN SENTIMENT ANALYSIS\Sarcasm_Headlines_Dataset.json\Sarcasm_Headlines_Dataset.json",lines=True )
df = df[['headline', 'is_sarcastic']]
df.head()


# Exploratory Data Analysis
# We would first like to understand the news headline dataset and identify the factors that causes a headline to be sarcastic news report. We would also like to identify if there is an imbalanced class, the frequencies and distribution of the different type of words. This will enable us to know if additional data processing steps are required (e.g. sampling of datapoints from sarcastic class if there is an imbalanced class in the dataset)
# 
# Data processing steps:
# 
# Check for missing values in headline, is_sarcastic
# Convert all words into lowercase
# Check for imbalanced classes in dataset
# Removal of punctuation

# In[5]:


# check for columns with null values
df.is_sarcastic.isnull().any() # no missing values in is_sarcastic column
df.headline.isnull().any() # no missing values in headline column


# In[ ]:





# In[ ]:





# In[6]:


df['headline'] = df.headline.apply(lambda x:x.lower())  # convert all words in headline into lower case 
df['headline'] = df.headline.apply(lambda x: ' '.join(word.strip(string.punctuation) for word in x.split()))  # remove all punct


# In[7]:


df['headline_count'] = df.headline.apply(lambda x: len(list(x.split())))
df['headline_unique_word_count'] = df.headline.apply(lambda x: len(set(x.split())))
df['headline_has_digits'] = df.headline.apply(lambda x: bool(re.search(r'\d', x)))
df


# In[8]:


sarcastic_dat = df.groupby('is_sarcastic').count()
sarcastic_dat.index = ['Non-sarcastic','Sarcastic']
plt.xlabel('Type of headlines (Sarcastic & Non-sarcastic)')
plt.ylabel('Frequencies of headlines')
plt.xticks(fontsize=10)
plt.title('Frequencies of Sarcastic vs Non-sarcastic headlines')
bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline_count)
bar_graph[1].set_color('r')
plt.show()


plt.xlabel('Type of headlines (Sarcastic & Non-sarcastic)')
plt.ylabel('Proportion of headlines')
plt.xticks(fontsize=10)
plt.title('Proportion of Sarcastic vs Non-sarcastic headlines')
bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline_count / sarcastic_dat.headline_count.sum())
bar_graph[1].set_color('r')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

# This is not an imbalanced class dataset
# Non-sarcastic    0.56
# Sarcastic        0.44
round(sarcastic_dat.headline_count / sarcastic_dat.headline_count.sum(), 2)


# In[9]:


all_dat = df.groupby('headline_count').count()
sarcastic_dat1 = df[df.is_sarcastic==1]
sarcastic_dat = sarcastic_dat1.groupby('headline_count').count()
not_sarcastic_dat1 = df[df.is_sarcastic==0]
not_sarcastic_dat = not_sarcastic_dat1.groupby('headline_count').count()

plt.xlabel('Different lengths of headline')
plt.ylabel('Frequencies of headline length')
plt.xticks(fontsize=10)
plt.title('Distribution of headline length for entire dataset')
bar_graph = plt.bar(all_dat.index, all_dat.headline)
bar_graph[8].set_color('r')
plt.axvline(df.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline
plt.show()

plt.xlabel('Different lengths of sarcastic headline')
plt.ylabel('Frequencies of sarcastic headline length')
plt.xticks(fontsize=10)
plt.title('Distribution of headline length for sarcastic dataset')
bar_graph = plt.bar(sarcastic_dat.index, sarcastic_dat.headline)
bar_graph[7].set_color('r')
plt.axvline(sarcastic_dat1.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline
plt.show()


plt.xlabel('Different lengths of non-sarcastic headline')
plt.ylabel('Frequencies of non-sarcastic headline length')
plt.xticks(fontsize=10)
plt.title('Distribution of headline length for non-sarcastic dataset')
bar_graph = plt.bar(not_sarcastic_dat.index, not_sarcastic_dat.headline)
bar_graph[8].set_color('r')
plt.axvline(not_sarcastic_dat1.headline_count.mean(), color='k', linestyle='dashed', linewidth=1)  # median is 10 words in a headline
plt.show()

# difference in the length of sarcastic and non-sarcastic headlines is not significant. 
# median and mean length of headlines is around 10 words


# In[10]:


digits_dat = df.groupby('headline_has_digits').count()
digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']


plt.xlabel('Type of headlines')
plt.ylabel('Frequencies of headlines')
plt.xticks(fontsize=10)
plt.title('Frequencies of headlines with Numbers vs No numbers')
bar_graph = plt.bar(digits_dat.index, digits_dat.headline / digits_dat.headline_count.sum())
bar_graph[1].set_color('r')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


sarcastic_digits_dat = df[df.is_sarcastic==1].groupby('headline_has_digits').count()
sarcastic_digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']


plt.xlabel('Type of headlines')
plt.ylabel('Frequencies of headlines')
plt.xticks(fontsize=10)
plt.title('Frequencies of Sarcastic headlines with Numbers vs No numbers')
bar_graph = plt.bar(sarcastic_digits_dat.index, sarcastic_digits_dat.headline / sarcastic_digits_dat.headline_count.sum())
bar_graph[1].set_color('r')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


not_sarcastic_digits_dat = df[df.is_sarcastic==0].groupby('headline_has_digits').count()
not_sarcastic_digits_dat.index = ['Has Numbers in Headline','Does not have Numbers in Headline']


plt.xlabel('Type of headlines')
plt.ylabel('Frequencies of headlines')
plt.xticks(fontsize=10)
plt.title('Frequencies of Non-sarcastic headlines with Numbers vs No numbers')
bar_graph = plt.bar(not_sarcastic_digits_dat.index, not_sarcastic_digits_dat.headline / not_sarcastic_digits_dat.headline_count.sum())
bar_graph[1].set_color('r')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

print(round(digits_dat.headline / digits_dat.headline_count.sum(),2))
print(round(sarcastic_digits_dat.headline / sarcastic_digits_dat.headline_count.sum(),2))
print(round(not_sarcastic_digits_dat.headline / not_sarcastic_digits_dat.headline_count.sum(),2))

# difference in the use of numbers/statistics in sarcastic and non-sarcastic headlines is not significant. 
# ~85% headlines uses numbers


# Identifying Topics in dataset via LDA
# Background
# LDA is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. Each document is modeled as a multinomial distribution of topics and each topic is modeled as a multinomial distribution of words. LDA assumes that the every chunk of text we feed into it will contain words that are somehow related. Therefore choosing the right corpus of data is crucial. It also assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. LDA assumes that each document (i.e. headline) consists of a mixture of topics (multinomial distribution) and each topic consists of a mixture of words (multinomial distribution).
# 
# Parameters
# LDA (short for Latent Dirichlet Allocation) is an unsupervised machine-learning model that takes documents as input and finds topics as output. The model also says in what percentage each document talks about each topic. There are 3 main parameters of the model:
# 
# the number of topics
# the number of words per topic
# the number of topics per document
# Data processing
# Before we can perform LDA, we will need to process the text in the following steps:
# 
# Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
# Removal of stop words.
# Removing Headlines that contains very few words. This helps to reduce the likelihood of headlines (that comprise of very but commonly-used words) matching together.
# Lemmatization: words in third person are changed to first person and verbs in past and future tenses are changed into present.
# Stemming: words are reduced to their root form.
# Whilst labelling each clusters give us an intuition of the meaning of each cluster, it is not necessary as the goal is not to label each document/headline into a cluster, but to measure the similarity between 2 documents/headlines. We can do so simply using similarity measures like Jensen-Shanon distance matric.
# 
# Limitations
# Need to pre-specify number of topics/clusters in advance.
# Heuristics to determine the optimal number of topics/clusters is largely based on domain knowledge and human interpretability.
# Cannot capture correlations between topics/clusters.

# In[11]:


import en_core_web_sm
nlp = en_core_web_sm.load()
parser = English()
en_stop = set(nltk.corpus.stopwords.words('english'))


def tokenize(text):
    """this function is to tokenize the headline into a list of individual words"""
    lda_tokens = []
    tokens = parser(text)  # need to use parser for python to treat the list as words
    for token in tokens:
        if token.orth_.isspace():  # to ignore any whitespaces in the headline, so that token list does not contain whitespaces 
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)   # tokens (headlines) are already in lowercase
    return lda_tokens


def get_lemma(word):
    """this function is to lemmatize the words in a headline into its root form"""
    lemma = wn.morphy(word)  # converts the word into root form from wordnet
    if lemma is None:
        return word
    else:
        return lemma
    

def prepare_text_for_lda(text):
    tokens = tokenize(text)  # parse and tokenize the headline into a list of words
    tokens = [token for token in tokens if len(token) > 4]  # remove headlines with only length of 4 words or less
    tokens = [token for token in tokens if token not in en_stop]  # remove stopwords in the headline
    tokens = [get_lemma(token) for token in tokens]  # lemmatize the words in the headline
    return tokens


# In[12]:


import nltk
nltk.download('stopwords')


# In[13]:


nltk.download('wordnet')


# In[14]:


text_data = []
for headline in df.headline:
    tokens = prepare_text_for_lda(headline)
    text_data.append(tokens)


# In[15]:


from gensim import corpora
import pickle

dictionary = corpora.Dictionary(text_data)  # Convert all headlines into a corpus of words, with each word as a token
corpus = [dictionary.doc2bow(text) for text in text_data]  # Convert each headline (a list of words) into the bag-of-words format. (Word ID, Count of word)
pickle.dump(corpus, open('corpus.pkl', 'wb'))  
dictionary.save('dictionary.gensim')  # takes a while to run the dictionary and corpus


# In[16]:


import gensim

NUM_TOPICS = [3, 5, 10]
# passes: Number of passes through the corpus during training
# alpha: priori on the distribution of the topics in each document.
# The higher the alpha, the higher the likelihood that document contains a wide range of topics, vice versa. 
# beta: priori on the distribution of the words in each topic.
# The higher the beta, the higher the likelihood that topic contains a wide range of words, vice versa.
# we do not alter / fine tune the default values of alpha and beta
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS[1], id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=5)
topics


# In[17]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
ldamodel.save('model3.gensim')
topics = ldamodel.print_topics(num_words=5)
topics


# In[18]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=5)
topics


# In[19]:


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)


# In[20]:


lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')
lda_display3 = pyLDAvis.gensim.prepare(lda3, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display3)


# In[21]:


lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display10)


# In[22]:


from numpy import mean

sarcastic = list(df.is_sarcastic == 1)
tuple_list = []
for headline in sarcastic:
    sarcastic = lda10[corpus[headline]]
    for tuple_ in sarcastic:
        tuple_list.append(tuple_)

print('For LDA model with 10 clusters:')
print('\nFor Sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

not_sarcastic = list(df.is_sarcastic == 0)
tuple_list = []
for headline in not_sarcastic:
    not_sarcastic = lda10[corpus[headline]]
    for tuple_ in not_sarcastic:
        tuple_list.append(tuple_)
        

print('\nFor Non-sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

# LDA model with 10 clusters not differentiable between sarcastic and not sarcastic headlines.
# Not very interpretable


# In[23]:


sarcastic = list(df.is_sarcastic == 1)
tuple_list = []
for headline in sarcastic:
    sarcastic = lda[corpus[headline]]
    for tuple_ in sarcastic:
        tuple_list.append(tuple_)

print('For LDA model with 5 clusters:')
print('For Sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

not_sarcastic = list(df.is_sarcastic == 0)
tuple_list = []
for headline in not_sarcastic:
    not_sarcastic = lda[corpus[headline]]
    for tuple_ in not_sarcastic:
        tuple_list.append(tuple_)
        

print('\nFor Non-sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

# LDA model with 5 clusters not differentiable between sarcastic and not sarcastic headlines.
# Not very interpretable


# In[24]:


sarcastic = list(df.is_sarcastic == 1)
tuple_list = []
for headline in sarcastic:
    sarcastic = lda3[corpus[headline]]
    for tuple_ in sarcastic:
        tuple_list.append(tuple_)

print('For LDA model with 3 clusters:')
print('For Sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

not_sarcastic = list(df.is_sarcastic == 0)
tuple_list = []
for headline in not_sarcastic:
    not_sarcastic = lda3[corpus[headline]]
    for tuple_ in not_sarcastic:
        tuple_list.append(tuple_)
        

print('\nFor Non-sarcastic Dataset:')
print([(uk, mean([vv for kk,vv in tuple_list if kk==uk])) for uk in set([k for k,v in tuple_list])])

# LDA model with 3 clusters not differentiable between sarcastic and not sarcastic headlines.
# Not very interpretable


# Conclusion of insights from EDA:
# What are the frequencies of sarcastic headlines against the non-sarcastic headlines?
# This is not an imbalanced class dataset. 56% of the headlines are non-sarcastic and 44% of the headlines are sarcastic. Hence, there is no requirements for conduct sampling to ensure an equal proportion of datasets from each class.
# 
# What is the word length of headlines? For sarcastic and non-sarcastic headlines?
# In the entire dataset, the mean and median length of headlines is around 10 words. There are some headlines with 2 / 3 / 4 words. These headlines need to be removed as they are too short. Short headlines will have a higher likelihood of being similar to other headlines without providing meaningful information in the topics.
# 
# There is no significant difference in the length of headlines for sarcastic and non-sarcastic datatset. </span>
# 
# Does the sarcastic headlines uses statistics (digits/numbers) in their wording? Compared to non-sarcastic headlines?
# In the entire dataset, 85% of all the headlines uses numbers. There is no significant difference in the use of numbers for sarcastic and non-sarcastic datatset. This suggests that most headlines uses numbers to attract viewership.
# 
# Topic Modelling. What are the topics in news headline for entire/sarcastic/non-sarcastic datasets?
# 
# What are all the topics in the news headline? Most popular topic?
# 
# For topic modelling, based on the dataset, the higher the number of topics, the more specialized the topics become. We have performed pre-processing of the data and subsequently applied LDA to create 3 / 5 / 10 topics. LDA for 10 topics helps us to understand the topics better. For example, for LDA10, topic 8 is generally about social media. Topic 9 is about politicians, namely Donald Trump. Topic 2 is about children, violence, climate change. For sarcastic and non-sarcastic datasets, the ratio of topics seems to be similar and there is no significant difference. The sizes of each topic is also similar. All these information suggests that sarcastic headlines is pervasive and appears throughout different genres of news.
# 
# What are all the topics involved in the news headline which are sarcastic? Most popular topic?
# What are all the topics involved in the news headline which are not sarcastic? Most popular topic?

# #### Prediction of Sarcasm in headlines using Deep Learning methods
# For the prediction of sarcasm of headlines, we will use RNN with the following architectures:
# 
# RNN with Gated Recurrent Units (with Lasso Regularization, Dropout, Batch Normalization)
# RNN with Gated Recurrent Units (with Ridge Regularization, Dropout, Batch Normalization)
# RNN with Long Short Term Memory Units (with Lasso Regularization, Dropout, Batch Normalization)
# RNN with Long Short Term Memory Units (with Ridge Regularization, Dropout, Batch Normalization)
# CNN with Conv1D
# Combination of CNN-RNN (LSTM)
# Output from CNN with Conv1D is used as input for RNN with LSTM (with Lasso Regularization, Dropout, Batch Normalization)

# #### LET US USE TRASNFORMER-BASED MODEL

# In[28]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# In[29]:


sentences = data['clean_headline']
label = data['is_sarcastic']


# #### Train - Validation - Test Splitting (80:10:10)

# In[30]:


X_train, X_val, y_train, y_val = train_test_split(sentences, label, test_size=0.2, stratify=label, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)


# ##### Tokenization
# Splitting sentences into words
# Finding the vocab size

# In[31]:


max_len = 20       
oov_token = '00_V' 
padding_type = 'post'
trunc_type = 'post'  

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: ",vocab_size)


# #### Encoding of Inputs
# Converting the sentences to token followed by padded sequences in encoded format
# These are numeric encodings assigned to each word

# In[32]:


train_sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences(X_val)
X_val = pad_sequences(val_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)


# #### Transformers: Attention is all you need
# For the purpose of classification problem we will only use the encoder block of the original transformers model (encoder-decoder) designed for sequence problems

# #### Multi-Headed Attention
# Multi-head Attention is a module for attention mechanisms which runs through an attention mechanism several times in parallel. The independent attention outputs are then concatenated and linearly transformed into the expected dimension.
# The Self Attention mechanism (illustrated in picture above next to the picture of encoder block) is used several times in parallel in Multi-Head attention
# Multiple attention heads allows for attending to parts of the sequence differently
# During self attention a word's attention score with itself will be the highest, therefore by using mutli-head attention a word can establish its relationship with other words in the sequence by calculating the attention scores with them in parallel

# In[33]:


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(neurons, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.5)
        self.dropout2 = layers.Dropout(0.5)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# #### Model Definition

# In[34]:


embed_dim = 50  
heads = 2  
neurons = 32
maxlen = 20
vocab_size = 20886

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerEncoder(embed_dim, heads, neurons)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = Dropout(0.35)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)


# In[35]:


model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# #### MODEL TRAINING 

# In[36]:


model_name = "model.h5"
checkpoint = ModelCheckpoint(model_name,
                            monitor="val_loss",
                            mode="min",
                            save_best_only = True,
                            verbose=1)

earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0.001, patience = 1, verbose = 1)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00000001)


# In[37]:


history = model.fit(X_train,y_train,
                    validation_data=(X_val,y_val),
                    epochs=25,
                    batch_size=32,
                    callbacks=[earlystopping])


# #### Model Evaluation
# ##### Learning Curves
# Loss Curve
# Accuracy Curve

# In[38]:


plt.figure(figsize=(20,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[39]:


plt.figure(figsize=(20,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# ##### Inference: In case of overfitting use the traditional methods
# 
# ###### Low learning rate
# ##### L1-L2 Regularization
# ##### Dropout
# ##### Lesser Neurons in MLP layers
# ##### Early Stopping
# etc.

# ##### Classification Metrics
# Since it is important to not misclassify the genuine headlines as sarcastic headlines we will also look at the roc auc score to avoid misclassification of genuine headlines as sarcastic headlines

# In[40]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, PrecisionRecallDisplay


# #### ROC Curve

# In[41]:


y_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="auc="+str(auc),lw=2)
plt.plot([0, 1], [0, 1], color="orange", lw=2, linestyle="--")
plt.legend(loc=4)
plt.show()


# ##### Scores: Test Set Result

# In[42]:


y_pred[y_pred>=0.85] = 1
y_pred[y_pred<0.85] = 0

print(classification_report(y_test, y_pred))


# #### CONFUSION MATRIX

# In[43]:


plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='.4g',cmap='viridis')


# In[ ]:





# In[ ]:





# ### RNN with GRU

# In[48]:


train_data, test_data = train_test_split(data[['headline', 'is_sarcastic']], test_size=0.1)  # randomly splitting 10% of dataset to be training dataset 

training_sentences = list(train_data['headline'])
training_labels = list(train_data['is_sarcastic'])

testing_sentences = list(test_data['headline'])
testing_labels = list(test_data['is_sarcastic'])
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


# In[49]:


vocab_size = 10000   # limit vector of words to the top 10,000 words
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

# no lemmatization, removal of stop words and stemming of headlines as we would like to maintain the syntax, literature integrity, sequence of words in LSTM.


# In[50]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[51]:


#Model Definition with BiRNN (GRU)
# with L1 Lasso Regularization, for feature selection
# Dropout, for robustness of recurrent neural networks
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_1 = tf.keras.Sequential([
   tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
   tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
   tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dropout(0.5),
   tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')
])
model_1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_1.summary()


# In[52]:


num_epochs = 10
history_1 = model_1.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[53]:


import matplotlib.pyplot as plt


def plot_graphs(history_1, string):
    plt.plot(history_1.history[string])
    plt.plot(history_1.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history_1, 'accuracy')
plot_graphs(history_1, 'loss')
plt.show()


# In[55]:


y_pred = model_1.predict(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_pred)
auc = roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="auc="+str(auc),lw=2)
plt.plot([0, 1], [0, 1], color="orange", lw=2, linestyle="--")
plt.legend(loc=4)
plt.show()


# In[ ]:


y_pred[y_pred>=0.85] = 1
y_pred[y_pred<0.85] = 0

print(classification_report(y_test, y_pred))


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='.4g',cmap='viridis')


# In[31]:


# Model Definition with BiRNN (GRU)
# with L2 Ridge Regularization
# Dropout, for robustness of recurrent neural networks
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')
])
model_2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_2.summary()


# In[32]:


num_epochs = 10
history_2 = model_2.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[33]:


plot_graphs(history_2, 'accuracy')
plot_graphs(history_2, 'loss')
plt.show()


# RNN with LSTM Architecture
# For the prediction of sarcasm of headlines, we will use RNN with the following architectures:
# 
# RNN with Long Short Term Memory Units (with Lasso Regularization, Dropout, Batch Normalization)
# RNN with Long Short Term Memory Units (with Ridge Regularization, Dropout, Batch Normalization)

# In[34]:


# Model Definition with BiRNN (LSTM)
# with L1 Lasso Regularization, for feature selection
# Dropout, for robustness of recurrent neural networks
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_3 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')
])

model_3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_3.summary()


# In[35]:


num_epochs = 10
history_3 = model_3.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[36]:


plot_graphs(history_3, 'accuracy')
plot_graphs(history_3, 'loss')
plt.show()


# In[37]:


# Model Definition with BiRNN (LSTM)
# with L2 Ridge Regularization
# Dropout, for robustness of recurrent neural networks
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_4 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')
])

model_4.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_4.summary()


# In[38]:


num_epochs = 10
history_4 = model_4.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[39]:


plot_graphs(history_4, 'accuracy')
plot_graphs(history_4, 'loss')
plt.show()


# In[40]:


# Model Definition with CNN (Conv1D)
# with L1 Lasso Regularization, for feature selection
# Dropout, for robustness
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_5 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.003), activation='sigmoid')
])
model_5.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_5.summary()


# In[41]:


num_epochs = 10
history_5 = model_5.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[42]:


plot_graphs(history_5, 'accuracy')
plot_graphs(history_5, 'loss')
plt.show()


# CNN
# For the prediction of sarcasm of headlines, we will use CNN with the following architectures:
# 
# CNN with Conv1D

# In[43]:


# Model Definition with CNN (Conv1D)
# with L2 Ridge Regularization
# Dropout, for robustness
# Batch Normalization, to stabilize and perhaps accelerate the learning process

model_6 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='sigmoid')
])
model_6.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_6.summary()


# In[44]:


num_epochs = 10
history_6 = model_6.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[45]:


plot_graphs(history_6, 'accuracy')
plot_graphs(history_6, 'loss')
plt.show()


# CNN-RNN combined architecture
# For the prediction of sarcasm of headlines, we will use the following architecture:
# 
# Combination of CNN-RNN (LSTM)
# Output from CNN with Conv1D is used as input for RNN with LSTM (with Lasso Regularization, Dropout)

# In[46]:


# Model Definition with CNN (Conv1D)
model_7 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 1, activation='relu'),
    tf.keras.layers.MaxPooling1D(2, padding="same"),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l1(0.005), activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.005), activation='sigmoid')
])
model_7.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_7.summary()


# In[47]:


num_epochs = 10
history_7 = model_7.fit(padded, training_labels_final, epochs=num_epochs, batch_size=64, validation_data=(testing_padded, testing_labels_final))


# In[48]:


plot_graphs(history_7, 'accuracy')
plot_graphs(history_7, 'loss')
plt.show()


# In[50]:


#plotting comparison between 4 models
import pandas as pd
from pandas import DataFrame
accuracy = [max(history_1.history['val_accuracy']),max(history_2.history['val_accuracy']), max(history_3.history['val_accuracy']),max(history_4.history['val_accuracy']),max(history_5.history['val_accuracy']),max(history_6.history['val_accuracy']),max(history_7.history['val_accuracy'])]
loss = [max(history_1.history['val_loss']),max(history_2.history['val_loss']),max(history_3.history['val_loss']),max(history_4.history['val_loss']),max(history_5.history['val_loss']),max(history_6.history['val_loss']),max(history_7.history['val_loss'])]

col={'Accuracy':accuracy,'Loss':loss}
models=['model_1','model_2','model_3','model_4','model_5','model_6','model_7']
graph_df=DataFrame(data=col,index=models)
graph_df


# In[51]:


graph_df.plot(kind='bar')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




