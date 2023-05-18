import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", engine="python")

df.head()

df.shape

tfidf = TfidfVectorizer(stop_words="english")

df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna('')

tfidf_matrix = tfidf.fit_transform(df["overview"])
tfidf_matrix.shape

tfidf.get_feature_names()

tfidf_matrix.toarray()

### Cosine Sim Calculator ####

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape


#### Recommendation Based on Similarities  ####

indices = pd.Series(df.index, index=df['title'])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep='last')]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df['title'].iloc[movie_indices]


##### Preparation of Working Script ######



def content_based_recommender(title, cosine_sim, dataframe):
    # creating indexes
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # capturing the index of 'title'
    movie_index = indices[title]
    # calculating similarity scores based on 'title'
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # retrieving the top 10 movies excluding itself.
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)