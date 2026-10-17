import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da página do Streamlit
st.set_page_config(page_title="Previsão de Consumo Industrial", layout="wide")

st.title("Painel de Previsão de Consumo de Energia Industrial")
st.markdown("""
Este painel utiliza Machine Learning (**Regressão Linear**) para prever o consumo de energia com base em parâmetros operacionais de uma fábrica.
""")

# 1. Geração do Dataset Simulado
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    n_registros = 40  # Mais de 20 registros conforme solicitado
    
    velocidade = np.random.uniform(1.0, 10.0, n_registros)
    maquinas = np.random.randint(1, 25, n_registros)
    temp = np.random.uniform(18.0, 38.0, n_registros)
    
    # Criando uma relação lógica para o consumo com um pouco de ruído aleatório
    consumo = (55.5 * velocidade) + (110.2 * maquinas) + (15.3 * temp) + np.random.normal(0, 35, n_registros)
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp,
        'Consumo_kWh': consumo
    })
    return df

df = gerar_dados()

# 2. Treinamento do Modelo de Regressão
X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
y = df['Consumo_kWh']

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Barra Lateral (Sidebar) com Controles Numéricos
st.sidebar.header("Parâmetros de Entrada")
st.sidebar.markdown("Ajuste os valores abaixo para simular o consumo:")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/s)", 
    float(X['Velocidade_Esteira'].min()), 
    float(X['Velocidade_Esteira'].max()), 
    float(X['Velocidade_Esteira'].mean())
)

maq_input = st.sidebar.slider(
    "Máquinas Ativas", 
    int(X['Maquinas_Ativas'].min()), 
    int(X['Maquinas_Ativas'].max()), 
    int(X['Maquinas_Ativas'].mean())
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)", 
    float(X['Temp_Ambiente'].min()), 
    float(X['Temp_Ambiente'].max()), 
    float(X['Temp_Ambiente'].mean())
)

# 4. Realizando a Previsão
entrada_usuario = np.array([[vel_input, maq_input, temp_input]])
consumo_previsto = modelo.predict(entrada_usuario)[0]

# 5. Exibição na Tela Principal
st.subheader("Resultado da Previsão")
st.metric(
    label="Consumo de Energia Estimado", 
    value=f"{consumo_previsto:.2f} kWh"
)

st.divider()

# 6. Gráfico de Contribuição das Variáveis (Pesos do Modelo)
st.subheader("Contribuição das Variáveis no Modelo")
st.markdown("O gráfico abaixo apresenta o **coeficiente (peso)** de cada variável no modelo de regressão, indicando o impacto relativo de cada fator no consumo:")

variaveis = ['Velocidade da Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
coeficientes = modelo.coef_

fig, ax = plt.subplots(figsize=(8, 4))
barras = ax.bar(variaveis, coeficientes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

ax.set_ylabel("Peso do Coeficiente")
ax.set_title("Importância dos Parâmetros na Regressão")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando os valores numéricos em cima de cada barra
for barra in barras:
    altura = barra.get_height()
    ax.annotate(f'{altura:.2f}',
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 4),  # Deslocamento vertical
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

st.pyplot(fig)

# Seção opcional para inspecionar os dados brutos
with st.expander("🔍 Visualizar Dataset de Treinamento"):
    st.dataframe(df, use_container_width=True)
