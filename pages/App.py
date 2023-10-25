import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ratings  = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\ratings_cleaned.csv",low_memory=False)

movies = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\movies.csv",low_memory=False)
links = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\links.csv",low_memory=False)

api_key = ""

st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🍿",
    layout="wide",
)

st.header('Movie Recommender System', anchor=None,divider=True)

#cleaning the titles to make getting the id easier 
movies['title_cleaned'] = movies['title'].astype(str).apply(lambda x: re.sub("[^a-zA-Z0-9 ]", "", x))

#creating a tfidf vectorizer to get the cosine similarity in the getid function
vect = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
movies_tfidf = vect.fit_transform(movies['title_cleaned'])


#function to get inputed moivies id
def getid(title):
    title =  re.sub("[^a-zA-Z0-9 ]", "", title)
    title_vector = vect.transform([title])
    title_cos_sim=cosine_similarity(title_vector, movies_tfidf).flatten()
    indices=np.argsort(title_cos_sim)[-10:]

    #HERER  
    movie_recommend = movies.iloc[indices][::-1]
    
    return movie_recommend

#function to recommend movies

@st.cache_data
def recommend(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .019]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["title", "genres","movieId"]]




def get_cover_img(movieId):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movieId}?api_key={api_key}')
    data = response.json()
    if "poster_path" in data and data["poster_path"]:
        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    else:
        return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"




#form to get input
with st.form(key='my_form'):
    movie_input=st.text_input("Enter a movie title:", key="recommend_input")


    if st.form_submit_button(label='Recommend'):

        data_load_state = st.text('Loading data...') 

        movie_id=getid(movie_input)
        movie_id = movie_id.iloc[0]["movieId"]  
        r_movies=recommend(movie_id)

        r_movies = pd.merge(r_movies, links, on='movieId', how='inner')
        st.divider()
    
        titles=r_movies["title"].tolist()  
        genres=r_movies["genres"].tolist()
        png=r_movies["tmdbId"].tolist()
    
        col1,col2,col3,col4,col5 = st.columns(5)
        cols=[col1,col2,col3,col4,col5]
        
        for i in range(0,5):
            with cols[i]:
                st.image(get_cover_img(png[i]), use_column_width=True)
                st.write(titles[i])
                st.write(genres[i])
    
        st.divider()
        col6,col7,col8,col9,col10 = st.columns(5)
        cols2=[col6,col7,col8,col9,col10]
    
        for i in range(5,10):
            with cols2[i-5]:
                st.image(get_cover_img(png[i]), use_column_width=True)
                st.write(titles[i])
                st.write(genres[i])
        data_load_state.text("✅ Here are your recommendations!")



# st.image(get_cover_img(862), use_column_width=True)
# movie_input=st.text_input("Enter a movie title:", key="recommend_input")


# if st.button("Recommend"):
    
#     movie_id=getid(movie_input)
#     movie_id = movie_id.iloc[0]["movieId"]  
#     r_movies=recommend(movie_id)
#     st.divider()

#     titles=r_movies["title"].tolist()  
#     genres=r_movies["genres"].tolist()

#     col1,col2,col3,col4,col5 = st.columns(5)
#     cols=[col1,col2,col3,col4,col5]
    
#     for i in range(0,5):
#         with cols[i]:
#             st.write(titles[i])
#             st.write(genres[i])

#     st.divider()
#     col6,col7,col8,col9,col10 = st.columns(5)
#     cols2=[col6,col7,col8,col9,col10]

#     for i in range(5,10):
#         with cols2[i-5]:
#             st.write(titles[i])
#             st.write(genres[i])
