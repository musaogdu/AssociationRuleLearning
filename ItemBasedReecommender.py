import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

df["title"].nunique()
df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

user_movie_df = common_movies.pivot_table(index= ["userId"], columns= ["title"], values = "rating")

user_movie_df.shape
user_movie_df.columns

#####  Making Movie Recommendations ####



movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)



def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Bad", user_movie_df)

###### Preparation of Working Script ########


def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()



def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

