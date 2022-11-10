import pickle
import pandas as pd
import streamlit as st

def preprocess_data(filename):
    # should return preprocessed data
    df = pd.read_csv(filename)

    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x == "Urban" else 0)

    df = pd.get_dummies(data=df, columns=['smoking_status'])
    df = pd.get_dummies(data=df, columns=['work_type'])

    df = df.dropna()

    x = df.drop('stroke', axis=1).values
    y = df["stroke"]
    return x, y

def create_model(model_file):
    model = None
    with open(model_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model
#
# model = create_model('rcv.pkl')
#
# x, y = preprocess_data('healthcare-dataset-stroke-data.csv')


st.title('Stroke detection')

st.checkbox('yes')
st.button('Click')
st.radio('Pick your gender',['Male','Female'])
st.selectbox('Pick your gender',['Male','Female'])
st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0,50)

st.write("Here's our first attempt at using data to create a table:")
st.dataframe(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

