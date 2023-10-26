import streamlit as st 
import json
from streamlit_lottie import st_lottie

def app():
    st.header('Welcome to the Movie Recommender System')
 

def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)

        
lottie_coding = load_lottiefile("./Animation/s2.json")
st_lottie(
            lottie_coding,
            speed =1,
            reverse = False,
            loop = True
        )


   

app()