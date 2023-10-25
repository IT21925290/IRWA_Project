#imports and libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading the data
ratings  = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\ratings_cleaned.csv",low_memory=False)
movies = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\movies.csv",low_memory=False)
links = pd.read_csv(r"C:\Users\nrhhe\Downloads\ml-25m\ml-25m\links.csv",low_memory=False)

#api key for the movie tmdb
api_key = "381a24ff761d56748f95b7e4e9b5a0c0"#please use your own api key(plz dont missuse mine)


st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🍿",
    layout="wide",
)

st.header('Movie Recommender System', anchor=None,divider=True)

#code to get the all the unique genres from the movies df to a list 
split = movies['genres'].str.split('|', expand=True)
unique_values = split[0].unique()


#function to get the genres to include in the recommendations
def genre_filter(unique_values):
    include_list=[]
    include_list = st.multiselect("Select genres to include:",unique_values)
    return include_list


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
def get_movies(movie_id,include_list):

    users_alike = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    movies_alike = ratings[(ratings["userId"].isin(users_alike)) & (ratings["rating"] > 4)]["movieId"]
    movies_alike = movies_alike.value_counts() / len(users_alike)

    movies_alike = movies_alike[movies_alike > .019]
    users_all = ratings[(ratings["movieId"].isin(movies_alike.index)) & (ratings["rating"] > 4)]
    user_recom = users_all["movieId"].value_counts() / len(users_all["userId"].unique())
    r_percentages = pd.concat([movies_alike, user_recom], axis=1)
    r_percentages.columns = ["similar", "all"]
    
    r_percentages["score"] = r_percentages["similar"] / r_percentages["all"]
    r_percentages = r_percentages.sort_values("score", ascending=False)
    
    #combing with movies df to get the title and genres
    combined=r_percentages.merge(movies, left_index=True, right_on="movieId")[["title", "genres","movieId"]]

    #filtering by genres 
    include=combined[combined["genres"].str.contains('|'.join(include_list))] 
    
    #returning the top 10 movies
    return include.head(10)[["title", "genres","movieId"]]



#function to get the cover image from the tmdb api
def get_cover_img(movieId):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movieId}?api_key={api_key}')
    data = response.json()
    if "poster_path" in data and data["poster_path"]:
        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    else:
        #catch errors if no poster is available and display a placeholder image
        return "https://via.placeholder.com/500x750.png?text=No+Poster+Available"




#form to get input
with st.form(key='my_form'):

    #getting users input
    movie_input=st.text_input("Enter a movie title:", key="recommend_input")

    #getting the genres to include
    include_list = genre_filter(unique_values)

    if st.form_submit_button(label='Recommend'):

        data_load_state = st.text('Loading data...') 
        
        #getting the movie id from the getid function
        movie_id=getid(movie_input)
        movie_id = movie_id.iloc[0]["movieId"]  

        #getting the recommendations from the get_movies function
        r_movies=get_movies(movie_id,include_list)

        #merging the links df to get the tmdbId and pass it to the tmdb api
        r_movies = pd.merge(r_movies, links, on='movieId', how='inner')
        st.divider()

        #getting the titles,genres and tmdbId from the df to lists
        titles=r_movies["title"].tolist()  
        genres=r_movies["genres"].tolist()
        png=r_movies["tmdbId"].tolist()

        #creating columns to display the recommendations
        col1,col2,col3,col4,col5 = st.columns(5)
        cols=[col1,col2,col3,col4,col5]

        #displaying the 1st 5 recommendations through looping 
        for i in range(0,5):
            with cols[i]:
                if i < len(titles):#catching any index errors
                    st.image(get_cover_img(png[i]), use_column_width=True)
                    st.write(titles[i])
                    st.write(genres[i])
                else:
                    st.write("No movies to show")
    
        st.divider()
        #creating columns to display the recommendations
        col6,col7,col8,col9,col10 = st.columns(5)
        cols2=[col6,col7,col8,col9,col10]

        #displaying the next 5 recommendations through looping 
        for i in range(5,10):
            with cols2[i-5] :
                if i < len(titles):#catching any index errors
                    st.image(get_cover_img(png[i]), use_column_width=True)
                    st.write(titles[i])
                    st.write(genres[i])
                else:
                    st.write("No movies to show")

        data_load_state.text("✅ Here are your recommendations!")
