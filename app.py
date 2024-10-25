import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import plotly.express as px
from datetime import datetime, timedelta

# Obtém os dados da ação
def get_dados_acao(ticker, inicio, fim):
    dados = yf.download(ticker, start=inicio, end=fim)
    return dados['Adj Close']

# Calcula a volatilidade histórica
def volatilidade_historica(retornos, janela=22):
    volatilidades = []
    for i in range(len(retornos)):
        if i >= janela:
            volatilidade = np.std(retornos[i-janela:i]) * np.sqrt(252)
            volatilidades.append(volatilidade)
        else:
            volatilidades.append(np.nan)
    return pd.Series(volatilidades, index=retornos.index)

# Calcula a volatilidade pelo GARCH(1,1)
def volatilidade_garch(retornos):
    modelo = arch_model(retornos, vol='Garch', p=1, o=1, q=1, dist='normal')
    resultado = modelo.fit()
    return resultado.params['alpha[1]'], resultado.params['beta[1]'], resultado.conditional_volatility

# Configuração da página
st.set_page_config(layout="wide")

# Título da página
st.title("Volatilidade de Ações")

# Formulário para entrada de dados
with st.form("dados_acao"):
    ticker = st.text_input("Ticker da ação", value="PETR4.SA")
    hoje = datetime.now()
    data_alterada = hoje - timedelta(days=365)
    inicio = st.date_input("Data de início", value=data_alterada)
    fim = st.date_input("Data de fim", value=pd.to_datetime(hoje))
    submitted = st.form_submit_button("Calcular")


# Processamento dos dados
if submitted:
    dados = get_dados_acao(ticker, inicio, fim)
    retornos = dados.pct_change().dropna()
    
    # Cálculo da volatilidade histórica
    vol_historica = volatilidade_historica(retornos)
    media_retornos = retornos.mean()
    ultima_vol_historica = vol_historica.dropna().iloc[-1]
    
    # Cálculo da volatilidade pelo GARCH(1,1)
    alpha, beta, vol_garch = volatilidade_garch(retornos)
    
    # Exibição dos resultados
    st.subheader("Resultados")
    col1, col2 = st.columns(2)
    col1.metric("Volatilidade Histórica", f"{ultima_vol_historica:.2%}")
    col2.metric("Volatilidade GARCH(1,1)", f"Alpha: {alpha:.4f}, Beta: {beta:.4f}")
    st.metric("Volatilidade GARCH(1,1) - Volatilidade Condicional", f"{vol_garch.iloc[-1]:.2%}")

    # Plotagem dos gráficos
    st.subheader("Gráficos")

 # Gráfico de Volatilidade Histórica
    vol_historica_df = pd.DataFrame({
        'Data': vol_historica.index,
        'Volatilidade Histórica': vol_historica.values
    })
    fig_vol_historica = px.line(vol_historica_df, x='Data', y='Volatilidade Histórica')
    fig_vol_historica.add_hline(y=vol_historica.dropna().mean(), line_dash="dash", line_color="red", annotation_text="Média")
    fig_vol_historica.update_layout(title='Volatilidade Histórica',
                                    xaxis_title='Data',
                                    yaxis_title='Volatilidade (%)')
    st.plotly_chart(fig_vol_historica, use_container_width=True)
    # Gráfico de Volatilidade GARCH
    vol_garch_df = pd.DataFrame({
        'Data': vol_garch.index,
        'Volatilidade GARCH(1,1)': vol_garch.values
    })    
    
    # Gráfico de Retornos
    fig_retornos = px.line(x=retornos.index, y=retornos.values)
    fig_retornos.add_hline(y=retornos.mean(), line_dash="dash", line_color="red", annotation_text="Média")
    fig_retornos.update_layout(title='Retornos',
                              xaxis_title='Data',
                              yaxis_title='Retorno (%)')
    st.plotly_chart(fig_retornos, use_container_width=True)
    

    fig_vol_garch = px.line(vol_garch_df, x='Data', y='Volatilidade GARCH(1,1)')
    fig_vol_garch.add_hline(y=vol_garch.mean(), line_dash="dash", line_color="red", annotation_text="Média")
    fig_vol_garch.update_layout(title='Volatilidade GARCH(1,1)',
                                xaxis_title='Data',
                                yaxis_title='Volatilidade (%)')
    st.plotly_chart(fig_vol_garch, use_container_width=True)    
