import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

def app():

    st.title("User Analytics in the Telecommunication Industry")

    st.header("Data Analysis")
    # processed_tweets = pd.read_csv("../../data/economic_clean.csv")
    parent_dir =os.path.join( os.path.abspath(os.path.join(os.getcwd(), os.pardir)),"week 1")
    file_path=os.path.join(parent_dir, "data", "experience-analytics.csv")
    clean_csv = pd.read_csv(file_path)
    userdata_info=pd.read_csv(os.path.join(parent_dir, "data", "clean_user_info.csv"))
    # model_ready_tweets.clean_text = model_ready_tweets.clean_text.astype(str)

    st.header("I.Handsets")
    st.subheader("1.The Top 10 Handset Types")
    
    st.bar_chart(clean_csv["Handset Type"].value_counts().nlargest(10))
    st.subheader("2.The Top 5 Handset Manufacturers")
    st.bar_chart(clean_csv["Handset Manufacturer"].value_counts().nlargest(5))
    

    
    st.subheader("Highest Used app compared to the total consumption")
    st.area_chart(userdata_info[["Total UL/DL", "Gaming UL/DL"]])
    hist_data = [userdata_info["Total UL/DL"],userdata_info["Gaming UL/DL"]]
    group_labels = ["Total UL/DL","Gaming UL/DL"]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
app()
