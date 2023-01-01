import time
import os
import streamlit as st 
#import streamlit.components.v1 as stc

#load EDA
import pandas as pd
#import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#loading dataset
def load_data(data):
    df = pd.read_csv(data)
    return df


#des[0];

#creating similarity matrix
def vectorize_text(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)

    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

#recommendation system
def get_recommendations(new_des,cosine_sim_mat,df):
    course_indices = pd.Series(df.index, index=df['Article'])
    idx = course_indices[new_des]
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1], reverse=True)
    selected_article_ind = [i[0] for i in sim_scores]
   #selected_article_score  = [i[0] for i in sim_scores[1:]]

    result_df = df.iloc[selected_article_ind]
    #result_df['similarity_score'] = selected_article_score
    #final_recommendation = result_df[['Article','similarity_score','Recommended Article']]
    #return final_recommendation
    return result_df.head(10)

def main():

    st.title("Mental Health Help")

    menu = ["Home","Recommended Articles","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    
    df = load_data("C:/Users/Kartik Mathur/OneDrive/Desktop/ISHIKA/Mental-Health-Help-Recomendation-Engine--main/mdh-1.csv")
    
    def make_clickable(link):
        # target _blank to open new window
        # extract clickable text to display for your link
        text = link.split('=')[0]
        return f'<a target="_blank" href="{link}">{text}</a>'
    #des = df['Condition Identified']
    #search_term = []
    if choice =="Home":
        st.subheader("Home")    

        # TRAILER is the column with hyperlinks
        df['Condition Identified'] = df['Condition Identified'].apply(make_clickable)

        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        

    elif choice == "Recommended Articles":
        st.subheader("Recommended articles")
        cosine_sim_mat = vectorize_text(df['Article'])
        search_term = st.text_input("Search")
       #num_of_rec = st.sidebar.number_input("Number",1,30,2)
        if st.button("Recommend Articles"):
            if search_term is not None:
                try:
                    results = get_recommendations(search_term,cosine_sim_mat,df)
                except:
                    results = "Not Found"
                with st.spinner('Loading ...'):
                    time.sleep(1.33)
                
                results['Condition Identified'] = results['Condition Identified'].apply(make_clickable)

                st.write(results.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                
    else:
        st.subheader("About")
        st.text("Built with Streamlit")



if __name__=='__main__':
    main()
