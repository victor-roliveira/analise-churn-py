import streamlit as st
import pandas as pd
import pickle

# Carregar o modelo, o scaler e as colunas que foram salvos
with open('modelo_churn.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('colunas_modelo.pkl', 'rb') as f:
    model_columns = pickle.load(f)
def predict_churn(data):
    # Criar um DataFrame com os dados de entrada
    df_input = pd.DataFrame([data])
    
    # Aplicar One-Hot Encoding
    df_input_encoded = pd.get_dummies(df_input).reindex(columns=model_columns, fill_value=0)
    
    # Padronizar as colunas numéricas com o scaler carregado
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumeroDeChamadas']
    df_input_encoded[numeric_cols] = scaler.transform(df_input_encoded[numeric_cols])
    
    # Fazer a previsão de probabilidade
    prediction_proba = model.predict_proba(df_input_encoded)
    
    return prediction_proba[0][1] # Retorna a probabilidade de Churn (classe 1)

st.title('Previsão de Churn de Clientes 🤖')
st.write('Insira os dados do cliente para prever a probabilidade de cancelamento.')

st.sidebar.header('Dados do Cliente')

tenure = st.sidebar.slider('Tempo como Cliente (meses)', 0, 72, 12)
contract = st.sidebar.selectbox('Tipo de Contrato', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.sidebar.selectbox('Serviço de Internet', ['DSL', 'Fiber optic', 'No'])
monthly_charges = st.sidebar.number_input('Cobrança Mensal ($)', min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input('Cobrança Total ($)', min_value=0.0, max_value=10000.0, value=500.0)
num_chamadas = st.sidebar.slider('Número de Chamadas ao Suporte', 0, 10, 1)

dados_cliente = {
    'gender': 'Female', 'Partner': 'Yes', 'Dependents': 'No', 'PhoneService': 'No',
    'MultipleLines': 'No phone service', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'tenure': tenure, 'Contract': contract, 'InternetService': internet_service,
    'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    'NumeroDeChamadas': num_chamadas
}

if st.button('**Prever Risco de Churn**'):
    probabilidade_churn = predict_churn(dados_cliente)
    
    st.markdown('---')
    st.subheader(f'A probabilidade deste cliente cancelar é de: **{probabilidade_churn:.2%}**')
    
    if probabilidade_churn > 0.5:
        st.error('**Risco de Churn: ALTO** 🔴')
        st.warning('Ação de retenção recomendada!')
    else:
        st.success('**Risco de Churn: BAIXO** 🟢')