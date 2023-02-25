import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Create UI

#Title
st.header('World Happiness Report')

#Inputs
Standard_Error = st.number_input("Enter the Standard Error score")#The standard error of the happiness score.

Economy_GDP_per_Capita = st.number_input("Enter the Economy")#The extent to which GDP contributes to the calculation of the Happiness Score.

Family = st.number_input("Enter family score ")#The extent to which Family contributes to the calculation of the Happiness Score

Health_Life_Expectancy = st.number_input("Enter health score")#The extent to which Life expectancy contributed to the calculation of the Happiness Score

Freedom = st.number_input("Enter freedom score")#The extent to which Freedom contributed to the calculation of the Happiness Score

Trust_Government_Corruption = st.number_input("Enter trust score")#The extent to which Perception of Corruption contributes to Happiness Score.

Generosity = st.number_input("Enter generosity score")#The extent to which Generosity contributed to the calculation of the Happiness Score.

Dystopia_Residual = st.number_input("Enter dystopia residual")#The extent to which Dystopia Residual contributed to the calculation of the Happiness Score.

#Button is pressed
if st.button("Submit") :

    #load the model for prediction
    clf = joblib.load('World_Happiness_Report.pkl')

    #Store inputs in dataframe
    X = pd.DataFrame([[Standard_Error,Economy_GDP_per_Capita,Family,Health_Life_Expectancy,Freedom,Trust_Government_Corruption,
                       Generosity,Dystopia_Residual]],
                                              
                       columns = ['Standard Error', 'Economy (GDP per Capita)', 'Family',
                                  'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
                                  'Generosity', 'Dystopia Residual'])
    
    predict = clf.predict(X)[0]

    st.text(f"Happiness score is {predict} ")

