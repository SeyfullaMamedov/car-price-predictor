import streamlit as st

def about_us():
    st.title('About Us')
    st.write("""
    CPP (Car Price Predictor) app was created to predict car prices based on various features using 
    machine learning algorithms, which will try its best for giving you best prices to you for calculating your 
    budget for affording a car based on its details that you will provide.
    

    **Creators:**
    - Seyfulla Mamedov -  is the last year student of Bachelor of Computer Engineering Department at Istanbul Arel University.
        For this app he used web scraping from sites and converted those details to dataframe by going through several 
        process such as Business Understanding, Data Collecting, Data Cleaning, Data Preprocessing, Feature Engineering, 
        Model Selection by using parameters, cross validations etc..
    """)
    st.write("**Acknowledgments:** We would like to thank Streamlit and the open-source community for their contributions.")