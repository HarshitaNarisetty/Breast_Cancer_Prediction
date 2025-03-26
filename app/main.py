import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
    data = pd.read_csv(r"C:\Users\RGUKT\Desktop\Python\DataScience\Projects\Breat Cancer Prediction\data\data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {key: (value - X[key].min()) / (X[key].max() - X[key].min()) for key, value in input_dict.items()}
    return scaled_dict

def get_rader_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    for name, prefix in zip(['Mean Value', 'Standard Error', 'Worst Value'], ['_mean', '_se', '_worst']):
        fig.add_trace(go.Scatterpolar(
            r=[input_data[f'radius{prefix}'], input_data[f'texture{prefix}'], input_data[f'perimeter{prefix}'],
               input_data[f'area{prefix}'], input_data[f'smoothness{prefix}'], input_data[f'compactness{prefix}'],
               input_data[f'concavity{prefix}'], input_data[f'concave points{prefix}'], input_data[f'symmetry{prefix}'],
               input_data[f'fractal_dimension{prefix}']],
            theta=categories,
            fill='toself',
            name=name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    return fig

def add_prediction(input_data):
    model = pickle.load(open(r"C:\Users\RGUKT\Desktop\Python\DataScience\Projects\Breat Cancer Prediction\model\model.pkl", "rb"))
    scaler = pickle.load(open(r"C:\Users\RGUKT\Desktop\Python\DataScience\Projects\Breat Cancer Prediction\model\scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell Cluster Prediction")
    st.markdown(f"### The cell cluster is: {'🟢 Benign' if prediction[0] == 0 else '🔴 Malignant'}")

    st.write(f"**Probability of being Benign:** {model.predict_proba(input_array_scaled)[0][0]:.2f}")
    st.write(f"**Probability of being Malignant:** {model.predict_proba(input_array_scaled)[0][1]:.2f}")

def add_sidebar():
    st.sidebar.header("📊 Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [(f"{key.replace('_', ' ').title()}", key) for key in data.columns if key != 'diagnosis']
    input_dict = {key: st.sidebar.slider(label, 0.0, float(data[key].max()), float(data[key].mean())) for label, key in slider_labels}
    return input_dict

def main():
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .main-title { color: #007bff; font-size: 3rem; }
        .description { color: #555; font-size: 1.2rem; }
        </style>
    """, unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.markdown("<h1 class='main-title'>🔬 Breast Cancer Predictor</h1>", unsafe_allow_html=True)
        st.markdown("""
            <p class='description'>This app analyzes cell characteristics from biopsy samples and provides predictions using a machine learning model.
            It assists healthcare professionals in early detection and diagnosis. Note: This is a diagnostic support tool and not a substitute for medical advice.</p>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        radar_chart = get_rader_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        add_prediction(input_data)

if __name__ == "__main__":
    main()
