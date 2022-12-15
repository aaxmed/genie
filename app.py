
import pandas as pd
import numpy as np

s = pd.read_csv('./social_media_usage.csv')

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return (x)


ss = pd.DataFrame({
    'sm_li':clean_sm(s["web1h"]),
    'income':np.where(s["income"]>9, np.nan, s["income"]),
    'education':np.where(s["educ2"]>8, np.nan, s["educ2"]),
    'parent':np.where(s["par"]==1, 1,0),
    'marital': np.where(s["marital"]==1, 1,0),
    'female': np.where(s["gender"]==2, 1,0),
    'age': np.where(s["age"]>98, np.nan, s["age"])
}).dropna()

y = ss["sm_li"]

x = ss.drop('sm_li', axis = 1)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



x_train, x_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2,
                                                   random_state=123)

lr=LogisticRegression(class_weight = "balanced")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)



import streamlit as st
import altair as alt

st.image('GTownCampus.jpeg', caption='Georgetown MSBA LinkedIn User Prediction App | Architect: AAhmed')


st.title('Want to play a game?')
st.subheader('I bet you I can predict if you are a Linkdin User or Not!')
st.text('Tell me a few things about you and I will use my magic crystal ball to predict!')


gender_answer = st.selectbox(label="What gender do you identify as?",
options=("Male",
         "Female",
         "Other",
         "Do not know",
         "Skip"))


marital_answer = st.selectbox(label="What is your relationship status?",
options=("Married",
         "Living with a partner",
         "Divorced",
         "Separated",
         "Widowed",
         "Never been married",
         "Do not know",
         "Skip"))

                       
parent_answer = st.selectbox(label="Do you have any kids?",
options=("Yes", "No", "Do not know", "Skip"))

age_answer = st.number_input('How old are you?:', min_value=0, max_value =98, step=1)

income_answer = st.selectbox(
    'How much money do you make in a year?',
    ('Less than 10,000', '10,000 - 20,0000', '20,000-30,000', '30,000-40,000', '40,000-50,000', '50,000-75,000', '75,000-100,000', '100,000-150,000:Hiring?', '150,000+: Hire me!'))

education_answer = st.selectbox(label="What is your education level?",
options=(
            "Less than high school (Grades 1-8 or no formal schooling)",
            "High school almost (Grades 9-11 or 12 with NO diploma)",
            "High school graduate (HS diploma or GED)",
            "Some college, no degree (some community college)",
            "2-year associate degree from a college or university",
            "4-year college or university degree/Bachelors degree",
            "Some postgraduate or professional schooling, but no degree (e.g. some graduate school)",
            "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD) - You really like school huh?",))


st.text('Small interactive info chart before prediction.')
chart = alt.Chart(ss).mark_circle().encode(
x="age",
y="income",
color="education"). \
properties(title="Info Chart: Social Media Users- Education & Income").interactive()
st.altair_chart(chart, use_container_width=True)


if gender_answer == "Female":
    gender_answer = 1
else:
    gender_answer = 0
    

if parent_answer == "Yes":
    parent_answer = 1
else:
    parent_answer = 0
    
    
if marital_answer == "Married":
    marital_answer = 1
else:
    marital_answer = 0
    
    
if income_answer == "Less than 10,000":
    income_answer = 1
elif income_answer == "10,000-20,0000":
    income_answer = 2
elif income_answer == "20,000-30,0000":
    income_answer = 3
elif income_answer == "30,000-40,000":
    income_answer = 4
elif income_answer == "40,000-50,000":
    income_answer = 5
elif income_answer == "50,000-75,000":
    income_answer = 6
elif income_answer == "75,000-100,000":
    income_answer = 7
elif income_answer == "100,000-150,000: Hiring?":
    income_answer = 8
elif income_answer == "150,000+: Hire me!":
    income_answer = 9

    
if education_answer == "Less than high school (Grades 1-8 or no formal schooling)": 
    education_answer = 1
elif education_answer == "High school almost (Grades 9-11 or 12 with NO diploma)":
    education_answer = 2
elif education_answer == "High school graduate (HS diploma or GED)":
    education_answer = 3
elif education_answer == "Some college, no degree (some community college)":
    education_answer = 4
elif education_answer == "2-year associate degree from a college or university":
    education_answer = 5
elif education_answer == "4-year college or university degree/Bachelors degree":
    education_answer = 6
elif education_answer == "Some postgraduate or professional schooling, but no degree (e.g. some graduate school)":
    education_answer = 7
elif education_answer == "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD) - You really like school huh?":
    education_answer = 8


answer_data = pd.DataFrame({
    "income": [income_answer],
    "education":[education_answer],
    "parent": [parent_answer],
    "marital": [marital_answer],
    "female": [gender_answer],
    "age": [age_answer]
})

st.subheader('From what you told me, my magic ball says...')
if st.button('Click here!'):

    predicted_class = lr.predict(answer_data)

    probs = lr.predict_proba(answer_data)

    if predicted_class == 0:
        print(st.info(f"You are not a LinkedIn User!- Suggestion: Become one :)"))
    else:
        print(st.info(f"You are a LinkedIn User!"))

    st.info(f"The probability that you use LinkedIn is: {round (probs[0][1], 2)}")

st.subheader("So, did I win the bet?!")
st.text('Let me know what you think, email me at ama502@georgetown.edu!')
st.text('Thank you for playing!')

st.image('viz.png', caption='I did it! | Architect: AAhmed')

