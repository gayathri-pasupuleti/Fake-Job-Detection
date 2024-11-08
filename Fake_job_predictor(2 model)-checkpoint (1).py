#!/usr/bin/env python
# coding: utf-8

# In[1]:


2+4#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string


# In[2]:


# Reading Dataset
data=pd.read_csv(r"C:/Users/PASUPULETI GAYATHRI/OneDrive/Pictures/Desktop/JOB POST/Dataset/Dataset/fake_job_postings.csv")


# In[3]:


# Reading top 5 rows of our dataset
data.head()


# In[4]:


# To check the number of rows and column
data.shape


# In[5]:


data.columns


# In[6]:


# let us check the missing values in our dataset

data.isnull().sum()


# In[7]:


# Let us remove the columns which are not necessary
# We have droped salary range because 70% approx null value also job_id and other irrelvent columns because they does not have any logical meaning
data.drop(['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions'],axis=1,inplace = True)


# In[8]:


data.shape


# In[9]:


data.head()


# In[9]:


data.dtypes


# In[10]:


# Fill NaN values with blank space
# inplace=true to make this change in the dataset permanent
data.fillna(' ', inplace=True)


# In[11]:


#Create independent and Dependent Features
columns = data.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["fraudulent"]]
# Store the variable we are predicting 
target = "fraudulent"
# Define a random state 
state = np.random.RandomState(42)
X = data[columns]
Y = data["fraudulent"]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)
from imblearn.under_sampling import RandomUnderSampler  

under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(X, Y)

df1 = pd.DataFrame(X_res)
  
df3 = pd.DataFrame(y_res)
  
# the default behaviour is join='outer'
# inner join
  
result = pd.concat([df1, df3], axis=1, join='inner')
display(result)
data=result;


# In[ ]:


data.isnull().sum()
# data cleaning done


# # Explaratory Data Analysis

# In[14]:


# Checking for distribution of class label(percentages belonging to real class and percentages belonging to fraud class)
# in the data 1 indicates fraud post
# 0 indicating real post
# Plotting pie chart for the data
# function of Explode function: how the portion will appear (to understand change explode=(0,0.5))

labels = 'Fake', 'Real'
sizes = [data.fraudulent[data['fraudulent']== 1].count(), data.fraudulent[data['fraudulent']== 0].count()]
explode = (0, 0.1) 
fig1, ax1 = plt.subplots(figsize=(8, 6)) #size of the pie chart
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
       shadow=True, startangle=120) #autopct %1.2f%% for 2 digit precision
ax1.axis('equal')
plt.title("Proportion of Fraudulent", size = 7)
plt.show() 


# In[15]:


# we will try to see which country is posting most of the jobs
# Visualize job postings by countries
# we will use the location column for visualizing this data
# In location data is of type (country_name,state,city)
# we neeed to know the country wise data

def split(location):
    l = location.split(',')
    return l[0]

data['country'] = data.location.apply(split)
data['country']


# In[16]:


# this will give unique country values
data['country'].nunique()


# In[17]:


# top 10 country that post jobs 
data['country'].value_counts()[:11]


# In[18]:


# creating a dictionary(key-value pair) with top 10 country
country = dict(data.country.value_counts()[:11])
del country[' '] #deleting country with space values
plt.figure(figsize=(12,9))
plt.title('Country-wise Job Posting', size=20)
plt.bar(country.keys(), country.values()) #(xaxis,yaxis)
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Countries', size=10)


# In[19]:


country.keys()


# In[20]:


# visualizing jobs based on experience
experience = dict(data.required_experience.value_counts())
del experience[' ']
plt.figure(figsize=(12,9))
plt.bar(experience.keys(), experience.values())
plt.title('No. of Jobs with Experience')
plt.xlabel('Experience', size=10)
plt.ylabel('No. of jobs', size=10)
plt.xticks(rotation=35)
plt.show()


# In[21]:


# Task: This data is Inbalanced, it contains 95% of real jobs and only 5% fake jobs,but we can make it balance
# Try this out


# In[22]:


#Most frequent jobs
print(data.title.value_counts()[:10])


# In[23]:


#Titles and count of fraudulent jobs
# checking for most fake jobs based on title
print(data[data.fraudulent==1].title.value_counts()[:10])


# In[24]:


# For textual type data we will try to create word cloud 
# but before that we will try to create text combining all the data present in
# our database.
data['text'] = data['title']+' '+data['location']+' '+data['company_profile']+' '+data['description']+' '+data['requirements']+' '+data['benefits']+' '+data['industry']

del data['title']
del data['location']
del data['department']
del data['company_profile']
del data['description']
del data['requirements']
del data['benefits']
del data['required_experience']
del data['required_education']
del data['industry']
del data['function']
del data['country']
del data['employment_type']


# In[25]:


data.head()


# In[26]:


data.tail(10)


# **Understanding the common words used in the texts : Wordcloud**

# In[27]:


# we will plot 3 kind of word cloud
# 1st we will visualize all the words our data using the wordcloud plot
# 2nd we will visualize common words in real job posting
# 3rd we will visualize common words in fraud job posting
# join function is a core python function
from wordcloud import WordCloud
all_words = ''.join([text for text in data["text"]]) 


# In[28]:


wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(all_words)


# In[29]:


plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[30]:


# Common words in real job posting texts

real_post = ''.join([text for text in data["text"][data['fraudulent']==0]])
wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(real_post)


# In[31]:


plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[32]:


# Common words in fraud job posting texts

fraud_post = ''.join([text for text in data["text"][data['fraudulent'] == 1]])


# In[33]:


wordcloud = WordCloud(width = 800, height = 500, random_state=21, max_font_size=120).generate(fraud_post)


# In[34]:


plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ## Data *Preapration*

# In[35]:


# NLTK :: Natural Language Toolkit
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords


# In[36]:


print(stopwords.words("english"))


# In[ ]:


#loading the stopwords
stop_words = set(stopwords.words("english"))


# In[ ]:


#converting all the text to lower case
data['text'] = data['text'].apply(lambda x:x.lower())


# In[ ]:


#removing the stop words from the corpus
data['text'] = data['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_words)]))


# In[ ]:


data['text'][0]


# In[ ]:


data['text'][1000]


# In[ ]:


y=np.array(data['fraudulent'])


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
# Splitting dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(data.text, y, test_size=0.3)


# In[ ]:


# what does X-train and y_train contain
print(y_train)
print(X_train)


# In[ ]:


import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


from sklearn import preprocessing

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=10)
count_trans = tfidf.fit(X_train) # fit has to happen only on train data

# Dump the file
pickle.dump(tfidf, open(r"C:\Users\strydo\Desktop\PROJECTS SEICOM\FAKE JOB\tfidf.pkl", "wb"))
# Testing phase
tfidf = pickle.load(open(r"C:\Users\strydo\Desktop\PROJECTS SEICOM\FAKE JOB\tfidf.pkl", 'rb'))

# we use the fitted CountVectorizer to convert the text to vector
X_train_tfidf =tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Normalize Data
X_train_tfidf = preprocessing.normalize(X_train_tfidf)
print("Train Data Size: ",X_train_tfidf.shape)

#Normalize Data
X_test_tfidf = preprocessing.normalize(X_test_tfidf)
print("Test Data Size: ",X_test_tfidf.shape)



# # Model Building & evaluation

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


# <h2>Decision Tree Classifier</h2>

# In[ ]:


#instantiate a Decision Tree Classifier
dt = DecisionTreeClassifier()


# In[ ]:


#train the model 
# using X_train_dtm (timing it with an IPython "magic command")

get_ipython().run_line_magic('time', 'dt.fit(X_train_tfidf, y_train)')


# In[ ]:


# make class predictions for X_test_dtm
y_pred_class = dt.predict(X_test_tfidf)


# In[ ]:


# Model Accuracy
print("Classification Accuracy:", accuracy_score(y_test, y_pred_class))
print("Classification Report\n")
print(classification_report(y_test, y_pred_class))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred_class))



# In[ ]:


# Confusion Matrix

import seaborn as sn
cm = confusion_matrix(y_test,y_pred_class)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier  
RF= RandomForestClassifier(n_estimators= 100, criterion="entropy")
get_ipython().run_line_magic('time', 'RF.fit(X_train_tfidf, y_train)')


# In[ ]:


y_pred_RF = RF.predict(X_test_tfidf)
print(y_pred_RF)


# In[ ]:


accuracy_score(y_test, y_pred_RF)
print("Classification Accuracy:", accuracy_score(y_test, y_pred_RF))
print("Classification Report\n")
print(classification_report(y_test, y_pred_RF))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred_RF))


cm = confusion_matrix(y_test,y_pred_RF)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:




