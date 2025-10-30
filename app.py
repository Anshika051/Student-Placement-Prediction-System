import streamlit as st
import joblib
import pandas as pd

df=pd.read_csv("collegePlacement.csv")

model=joblib.load("placement_model.pkl")
le=joblib.load("labelencoder.pkl")
oh_encoder=joblib.load("encoder.pkl")

st.write("Student Placement Prediction")

Stream=st.selectbox("Select Your Stream",df['Stream'].unique().tolist())
Age=st.slider("Select your Age",min_value=19,max_value=30)
Gender=st.selectbox("Select Your Gender",df['Gender'].unique().tolist())
Internships=st.slider("Enter the number of internships done",min_value=0,max_value=8)
CGPA=st.slider("Enter your CGPA:",min_value=1,max_value=10)
Hostel=st.selectbox("You live in Hostel?",df['Hostel'].unique().tolist())
Backlogs=st.slider("Select the number of backlogs you have",min_value=0,max_value=5)

user_input={'Stream': Stream,
    'Age': Age,
    'Gender': Gender,
    'Internships': Internships,
    'CGPA': CGPA,
    'Hostel': Hostel,
    'HistoryOfBacklogs':Backlogs}

user_df=pd.DataFrame([user_input])

user_df['Gender']=le.transform(user_df['Gender'])
encoded_input=oh_encoder.transform(user_df)

st.divider()
if st.button("Predict Placement Probability"):
    pred=model.predict(encoded_input)
    probability=model.predict_proba(encoded_input)[0][1]
    st.write("Placement Prediction:", "Placed" if pred[0] == 1 else "Not Placed")
    st.write(f"Probability of placement: {probability:.2f}")


