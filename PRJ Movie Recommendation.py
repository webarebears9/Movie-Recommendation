#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics.pairwise import cosine_similarity
vec1 = [1,1,0,1,1]
vec2 = [0,1,0,1,1]
print(cosine_similarity([vec1, vec2]))


# In[2]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer - This is used to convert text data into numerical values
from sklearn.metrics.pairwise import cosine_similarity
import os


# In[3]:


os.chdir ("C:\\Users\\anilb\\OneDrive\\Desktop\\clg\\acmegrade assignments")


# In[4]:


display (os.getcwd())


# In[5]:


movies_data =pd.read_csv('movies.csv')
movies_data.head()


# In[6]:


display (movies_data.shape)


# In[7]:


selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[8]:


display (movies_data.info())


# In[9]:


display (movies_data.isna().sum())


# In[10]:


display (movies_data[selected_features].head())


# In[11]:


display (movies_data[selected_features].isna().sum())


# In[12]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
display (movies_data.head())


# In[13]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
display (combined_features)


# In[14]:


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
display (feature_vectors.shape)
print (feature_vectors)


# In[15]:


similarity = cosine_similarity(feature_vectors)
print  (similarity )


# In[16]:


display(similarity.shape)


# In[17]:


movie_name = input(' Enter your favourite movie name : ')


# In[18]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[19]:


len(list_of_all_titles)


# In[20]:


find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[21]:


close_match = find_close_match[0]
print(close_match)


# In[22]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[23]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[24]:


len(similarity_score)


# In[25]:


sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[26]:


print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1

