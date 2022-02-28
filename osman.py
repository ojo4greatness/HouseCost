
import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.write(""" AIRBNB Prediction App""")

model=pickle.load(open('model_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


st.sidebar.header('User Input References')

def user_input_features():
    host_id=st.number_input('What is the Host ID:', step=1)
    calculated_host_listings_count = st.number_input('Host listing count:', step=1)
    longitude = st.number_input("Building's longitudinal location:")
    latitude = st.number_input("Building's latitudinal location:")
    room_type = st.sidebar.selectbox('Room type',('Entire home/apt','Private room','Shared room'))
    
    if room_type == 'Entire home/apt':
        room_type_Private=0
        room_type_Shared=0
   
    if room_type == 'Private room':
        room_type_Private=1
        room_type_Shared=0
    else:
        room_type_Private=0
        room_type_Shared=1

    
    minimum_nights = st.number_input('How many nights will you be staying:',step=1)
    availability_365 = st.number_input('For how manuy days is the building available:',step=1, min_value=0, max_value=365)
    last_review_Month = st.number_input('On which month was the last review:',step=1, min_value=1, max_value=12)
    last_review_Year = st.number_input('On which Year was the last review:',step=1, min_value=2012, max_value=2022)
    
    reviews_per_month = st.number_input('How many reviews do you receive monthly:',step=1)
    number_of_reviews = st.number_input('Number of Reviews received:',step=1)
    neighbourhood_group = st.sidebar.selectbox('Neighbourhood Group',['North-East Region','East Region','West Region','North Region', 'Central Region'])     


    if neighbourhood_group == 'North-East Region':
        neighbourhood_group_North_East_Region=0
        neighbourhood_group_East_Region=0
        neighbourhood_group_West_Region=0
        neighbourhood_group_North_Region=0

    if neighbourhood_group == 'East Region':
        neighbourhood_group_North_East_Region=1
        neighbourhood_group_East_Region=0
        neighbourhood_group_West_Region=0
        neighbourhood_group_North_Region=0

    if neighbourhood_group == 'West Region':
        neighbourhood_group_North_East_Region=0
        neighbourhood_group_East_Region=1
        neighbourhood_group_West_Region=0
        neighbourhood_group_North_Region=0

    if neighbourhood_group == 'North Region':
        neighbourhood_group_North_East_Region=0
        neighbourhood_group_East_Region=0
        neighbourhood_group_West_Region=1
        neighbourhood_group_North_Region=0

    else:
        neighbourhood_group_East_Region=0
        neighbourhood_group_West_Region=0
        neighbourhood_group_North_Region=0
        neighbourhood_group_Central_Region=1
    
    data = {
        'host_id':host_id,
        'calculated_host_listings_count':calculated_host_listings_count,
        'longitude':longitude,
        'latitude':latitude,
        'minimum_nights':minimum_nights,
        'availability_365':availability_365,
        'Year':last_review_Year,
        'Month':last_review_Month,
        'reviews_per_month':reviews_per_month,
        'number_of_reviews':number_of_reviews,
        'room_type_Private room':room_type_Private,
        'room_type_Shared room':room_type_Shared,
        'neighbourhood_group_North-East Region':neighbourhood_group_North_East_Region,
        'neighbourhood_group_East Region':neighbourhood_group_East_Region,
        'neighbourhood_group_West Region':neighbourhood_group_West_Region,
        'neighbourhood_group_North Region':neighbourhood_group_North_Region

        

    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
input_df = scaler.transform(input_df)

if st.button('PREDICT'):
    y_out = model.predict(input_df)
    st.write(f' This room will cost you $', y_out[0])
    
#st.markdown('My Wonderful app')
#st.number_input('Choose your age',step=1, min_value=18, max_value=65)
#st.sidebar.select_slider('Choose your salary', (10000, 45000, 600000))
#st.sidebar.selectbox('Choose your State',['Enugu','Jigawa','Gombe'])
