import streamlit as st
import pandas as pd
import pickle
import base64
from sklearn.preprocessing import StandardScaler

# Load your trained model and preprocessing info
with open('churn_predict_best_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('preprocessing_info.pkl', 'rb') as file:
    preprocessing_info = pickle.load(file)

categorical_columns = preprocessing_info['categorical_columns']
expected_columns = preprocessing_info['columns']

def preprocess_data(data, preprocessing_info):
    # Drop unnecessary columns
    data = data.drop(columns=['customer_id', 'churn'], errors='ignore')

    # Create derived features
    if 'age' in data.columns:
        bins = [0, 30, 50, 100]
        labels = ['young', 'middle-aged', 'senior']
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

    # One-hot encode categorical columns
    data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Ensure the columns are in the correct order
    data_encoded = data_encoded[expected_columns]

    # Feature scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_encoded)

    return data_scaled

def get_user_input():
    st.sidebar.header("User Input Parameters")
    
    credit_score = st.sidebar.slider('Credit Score', min_value=300, max_value=850, value=600)
    age = st.sidebar.slider('Age', min_value=18, max_value=100, value=30)
    tenure = st.sidebar.slider('Tenure', min_value=0, max_value=10, value=5)
    balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=250000.0, value=50000.0)
    num_of_products = st.sidebar.selectbox('Number of Products', [1, 2, 3, 4], index=1)
    has_cr_card = st.sidebar.radio('Has Credit Card', [0, 1])
    is_active_member = st.sidebar.radio('Is Active Member', [0, 1])
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0)
    country = st.sidebar.selectbox('Country', ['France', 'Spain', 'Germany'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

    data = {'credit_score': credit_score,
            'age': age,
            'tenure': tenure,
            'balance': balance,
            'products_number': num_of_products,
            'credit_card': has_cr_card,
            'active_member': is_active_member,
            'estimated_salary': estimated_salary,
            'country': country,
            'gender': gender}
    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.title("Customer Churn Prediction Dashboard")

    # Create tabs
    tabs = st.tabs(["Home", "Predictions", "Upload Data"])

    with tabs[0]:
        st.header("Home")
        st.write("Welcome to the Customer Churn Prediction Dashboard.")

    with tabs[1]:
        st.header("Predictions")
        user_input = get_user_input()
        st.write("User Input Parameters:", user_input)
        
        # Preprocess user input data
        user_input_encoded = preprocess_data(user_input, preprocessing_info)
        
        prediction = model.predict(user_input_encoded)
        prediction_proba = model.predict_proba(user_input_encoded)
        st.write("Prediction (0 = No Churn, 1 = Churn):", prediction)
        st.write("Prediction Probability:", prediction_proba)
        
        # Adding a bar chart for prediction probabilities
        st.subheader('Prediction Probability Bar Chart')
        probabilities = prediction_proba[0]
        probability_df = pd.DataFrame(probabilities, index=model.classes_, columns=['Probability'])
        st.bar_chart(probability_df)

    with tabs[2]:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            uploaded_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:", uploaded_data.head())

            # Preprocess uploaded data
            uploaded_data_encoded = preprocess_data(uploaded_data, preprocessing_info)

            # Make predictions on uploaded data
            uploaded_predictions = model.predict(uploaded_data_encoded)
            uploaded_data['Prediction'] = uploaded_predictions
            uploaded_data['Prediction Probability'] = model.predict_proba(uploaded_data_encoded)[:, 1]
            
            st.write("Predictions:", uploaded_data)

            # Provide download link for predictions
            csv = uploaded_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            linko = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
