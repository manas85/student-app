import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open('student_lr_model.pkl','rb') as file:
        model,scaler,le = pickle.load(file)
    
    return model, scaler, le

def preprocessing_input_data ( data, scaler, le):
    
    data['Extracurricular Activities'] =le.fit_transform(data['Extracurricular Activities'])
    df_transformed = scaler.transform(data)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    preprocess_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(preprocess_data)
    return prediction[0]

def main():
    st.title('Student performance prediction')
    st.write('enter your detail to get prediction for your input')

    hours_studies = st.number_input("Hours Studied", min_value=1,max_value= 10, value= 5)
    previous_score =st.number_input("Previous Scores", min_value= 40 , max_value= 100, value=50)
    extracurricular_activities = st.selectbox("Extracurricular Activities",["Yes","No"])
    sleep_hour = st.number_input("Sleep Hours", min_value=4, max_value=10 , value=7)
    question_paper = st.number_input("Question Papers", min_value= 0, max_value= 10 , value= 5)


    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied": hours_studies,
            "Previous Scores": previous_score,
            "Extracurricular Activities":extracurricular_activities,
            "Sleep Hours":sleep_hour,
            "Sample Question Papers Practiced":question_paper

        }
        user_data_df = pd.DataFrame([user_data])
        prediction = predict_data(user_data_df)
        st.success(f"Your prediction result is: {prediction} ")

if __name__ == "__main__":
    main()