import streamlit as st
import pickle 
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32','id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1 , 'B':0})
    return data

def side_bar():
    st.sidebar.header("Cell Nuclie Measurements")
    data = clean_data()

    slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]



    input_dict = {}
    for label,key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= float(0),
            max_value= float(data[key].max()),
            value = float(data[key].mean())
        )

    return input_dict   


def get_scaled_data(input_dict):
    data = clean_data()
    x = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = max(data[key])
        min_val = min(data[key])
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict



def radar_chart(input_data):
    input_data = get_scaled_data(input_data)

    categories = ['Radius','Texture','Perimeter','Area','Smoothness',
                  'Compactness','Concavity','Concave points','Symmetry','Fractal dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],  
        theta=categories,
        fill='toself',
        name='Mean Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error Values'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Values'
    ))
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig
    
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl","rb"))
    scaled = pickle.load(open("model/scaler.pkl","rb"))

    input_array = np.array(list(input_data.values())).reshape(1,-1)
    input_scaled = scaled.transform(input_array)
    pred = model.predict(input_scaled)
    st.subheader("Cell Cluster Prediction")
    st.write("Cell cluster is:")
    
    if pred[0] == 1:
       st.write("<p class='Diagnosis mal'>Malignant<br>(cancerous)</p>",unsafe_allow_html=True)
    else:
        st.write("<p class='Diagnosis beg'>Benign<br>(non-cancerous)</p>",unsafe_allow_html=True)

    st.write("Probability of Malignant Tumor: ",model.predict_proba(input_scaled)[0][1])
    st.write("Probability of Benign Tumor: ",model.predict_proba(input_scaled)[0][0])
    st.write("Note: Thsis model can be used for assisting medical professionals . Always consult a medical professional for diagnosis and treatment options.")


def main():
    st.set_page_config(
        page_title= "Breast Cancer Predictor",
        page_icon= "🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = side_bar()

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This is an Machine Learning based Model.you can Use/connect this app to your cytology lab to help diagnose breast cancer from your tissue sample.The model analyzes medical data and predicts whether a tumor is benign (non-cancerous) or malignant (cancerous) based on input features derived from diagnostic tests.")

    col1 , col2 = st.columns([4,1])

    with col1:
        radar = radar_chart(input_data)
        st.plotly_chart(radar)
        st.write("<p class='chart'><b><u>Above Chart Description<u><b></p>", unsafe_allow_html=True)
        st.write("This is a radar chart that visualizes the measurements of various cell nuclei features. " 
        "The chart displays three sets of values: mean, standard error, and worst values for each feature. " 
        "The features include radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractional dimension. " 
        "Each set of values is represented by a different color in the chart.")

    with col2:
        add_predictions(input_data)

    # data = clean_data()
    # print("Available columns:", data.columns)



if __name__== '__main__':
     main()
    
