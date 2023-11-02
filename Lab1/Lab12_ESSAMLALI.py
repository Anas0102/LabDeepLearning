# Lab_12: classification des fleurs iris
# Réaliser par Anas ESSAMLALI EMSI 2023/2024

import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Step 1:Dataset
iris =datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
print(iris.data.shape)
# Step 2: Create a Model

model= RandomForestClassifier()

# Step 3: Train the Model

model.fit(iris.data,iris.target)


# Step 4: Test the Model

prediction = model.predict([[0.9,1.,2.1,1.8]])
print(iris.target_names[prediction])

#model deployment with streamlit :streamlit run name.py

st.header("Classification des images iris")

st.sidebar.header("iris features")
def user_input():
    Sepal_length = st.sidebar.slider("Sepal length,0.1,10,5")
    Sepal_width = st.sidebar.slider("Sepal width,0.1,10,5")
    petall_length = st.sidebar.slider("petal length,0.1,10,5")
    petal_width= st.sidebar.slider("petall width,0.1,10,5")

    data = {
        "Sepal_length":Sepal_length,
        "Sepal_width ":Sepal_width,
        "petall_length":petall_length,
        " petal_width": petal_width
    }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features
df=  user_input()
st.write(df)

st.subheader("Prediction")
prediction= model.predict(df)
st.write(prediction)
st.write(iris.target_names[prediction])
st.image("images/"+ iris.target_names[prediction][0]+ ".png")
selected_model= st.sidebar.selectbox("Select your model", ["RandomForest","DecisionTree","KNN","SVN"])
st.write("Selected model is :", selected_model)