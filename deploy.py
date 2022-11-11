import pickle
import pandas as pd
import streamlit as st


def preprocess_data(data):
    ohe = None
    with open('ohe.pkl', 'rb') as ohe_file:
        ohe = pickle.load(ohe_file)

    df_temp = ohe.transform(
        data[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]).toarray()

    encodings = pd.DataFrame(columns=ohe.get_feature_names_out(), data=df_temp)
    encodings = encodings.astype(int)
    data = pd.concat([data, encodings], axis=1)

    data.drop(['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis=1, inplace=True)

    x = data.values
    print(data.columns)
    return x


def create_model(model_file):
    model = None
    with open(model_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


model = create_model('rf.pkl')

st.title('Stroke detection')
st.write('Explain the project here')

st.subheader('Fill in your data')
upload_methode = st.selectbox('Select methode', ['Form', 'Dataset'])

user_data = None

if upload_methode == 'Form':
    st.write('Please fill in this form')

    gender = st.radio('Pick your gender', ['Male', 'Female'])
    age = st.number_input('How old are you?', step=1)
    st.write('Check the box if suffer any of these')
    hypertension = st.checkbox('hypertension')
    heart_disease = st.checkbox('heart disease')
    married = st.radio('Are you married', ['Yes', 'No'])
    work = st.selectbox('Choose your work type?', ['Private', 'Self-employed', 'Government job', 'children', 'Never worked'])
    residence = st.selectbox('Choose your residence type?', ['Urban', 'Rural'])
    avg_glucose_level = st.number_input('Average glucose level?', step=0.1)
    bmi = st.number_input('Body Mass Index', step=0.1)
    smoking_status = st.selectbox('Choose your smoking status', ['formerly smoked', 'never smoked', 'smokes'])

    user_data = pd.DataFrame({
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': married,
        'work_type': 'Govt_job' if work == 'Government job' else 'Never_worked' if work == 'Never worked' else work,
        'Residence_type': residence,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }, index=[0])

else:
    pass

# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Ba
# d', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)

st.write('User Data')

st.dataframe(user_data)

if st.button('Predict'):
    st.write(f'Model Prediction: {model.predict(preprocess_data(user_data))}')
