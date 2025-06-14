import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np # Adicionado para operações numéricas e seleção de tipos

st.set_page_config(layout="wide") # Movido para o início

# --------------------
#        Dados
# --------------------

# Cache data loading to improve performance
@st.cache_data
def load_data(file_path):
    df_loaded = pd.read_csv(file_path)
    #Eliminar todas as linhas com valores Nan
    df_loaded.dropna(axis=0, inplace=True)
    df_loaded.reset_index(drop=True, inplace=True)
    #Separando os valores de data e hora
    df_loaded['timestamp'] = pd.to_datetime(df_loaded['timestamp'])
    df_loaded['date'] = df_loaded['timestamp'].dt.date
    df_loaded['time'] = df_loaded['timestamp'].dt.time
    return df_loaded

df_original = load_data('smart_manufacturing_data.csv')

# --------------------
#       FILTROS STREAMLIT
# --------------------
st.sidebar.header("Filtros")

# Opções para filtros (baseadas no dataset original completo)
lista_maquinas_original = ['Todas'] + sorted(df_original['machine'].unique().tolist())
min_date_original = df_original['date'].min()
max_date_original = df_original['date'].max()
lista_tipos_falha_original = ['Todos'] + sorted(df_original['failure_type'].unique().tolist())

# Filtro de Máquina
maquinas_selecionadas = st.sidebar.multiselect(
    "Selecione a(s) Máquina(s):",
    options=lista_maquinas_original,
    default=['Todas']
)

# Filtro de Data
data_inicio_selecionada = st.sidebar.date_input("Data Início:", value=min_date_original, min_value=min_date_original, max_value=max_date_original)
data_fim_selecionada = st.sidebar.date_input("Data Fim:", value=max_date_original, min_value=min_date_original, max_value=max_date_original)

# Filtro por Tipo de Falha
tipos_falha_selecionados = st.sidebar.multiselect(
    "Selecione o(s) Tipo(s) de Falha:",
    options=lista_tipos_falha_original,
    default=['Todos']
)

# Aplicar filtros ao DataFrame
df = df_original.copy()

# --------------------------
#    Tratamento de Dados
# --------------------------
if data_inicio_selecionada > data_fim_selecionada:
    st.sidebar.error("Erro: Data de início não pode ser posterior à data de fim.")
    # Mantém o df com todos os dados ou o último estado válido em vez de parar
    # Ou você pode optar por st.stop() se preferir interromper a renderização
else:
    df = df[(df['date'] >= data_inicio_selecionada) & (df['date'] <= data_fim_selecionada)]

if not ('Todas' in maquinas_selecionadas or not maquinas_selecionadas):
    df = df[df['machine'].isin(maquinas_selecionadas)]

if not ('Todos' in tipos_falha_selecionados or not tipos_falha_selecionados):
    df = df[df['failure_type'].isin(tipos_falha_selecionados)]

# O restante do tratamento de dados e geração de tabelas/gráficos usará o 'df' filtrado.
# É importante adicionar verificações para 'df.empty' antes de cada operação de groupby ou plotagem
# para evitar erros se a combinação de filtros resultar em um DataFrame vazio.

# --------------------
#        TABELAS E CÁLCULOS DE CORRELAÇÃO
# --------------------

# Inicializar DataFrames de correlação como vazios
corr_pearson = pd.DataFrame()
corr_spearman = pd.DataFrame()
# Inicializar variáveis para métricas
maquina_mais_paradas = "N/A"
cont_mais_paradas = 0
maquina_menos_paradas = "N/A"
cont_menos_paradas = 0


if not df.empty:
    media_consumo_energia = df.groupby('machine')['energy_consumption'].mean().reset_index()
    mediatemp = df.groupby('machine')['temperature'].mean().reset_index()
    media_press = df.groupby('machine')['pressure'].mean().reset_index()
    media_vibration = df.groupby('machine')['vibration'].mean().reset_index()

    df_manutencao = df[df['maintenance_required'] == "Yes"]
    if not df_manutencao.empty:
        df_manutencao_required = df_manutencao.groupby('machine')['maintenance_required'].count().reset_index(name='contagem_manutencao')
    else:
        df_manutencao_required = pd.DataFrame(columns=['machine', 'contagem_manutencao'])

    rodando_count = df[df['machine_status'] == "Running"]['machine'].count()
    total_count_filtrado = df['machine'].count()
    perc_rodando = (rodando_count / total_count_filtrado) * 100 if total_count_filtrado > 0 else 0

    falha_count = df[df['machine_status'] == "Failure"]['machine'].count()
    perc_falha = (falha_count / total_count_filtrado) * 100 if total_count_filtrado > 0 else 0

    parada_count_total = df[df['machine_status'] == "Idle"]['machine'].count() # Renomeado para evitar conflito
    perc_parada = (parada_count_total / total_count_filtrado) * 100 if total_count_filtrado > 0 else 0

    # Cálculo para métricas de paradas por máquina
    df_paradas_maquina = df[df['machine_status'] == 'Idle']
    if not df_paradas_maquina.empty:
        contagem_paradas_por_maquina = df_paradas_maquina.groupby('machine')['machine_status'].count().sort_values(ascending=False)
        if not contagem_paradas_por_maquina.empty:
            maquina_mais_paradas = contagem_paradas_por_maquina.index[0]
            cont_mais_paradas = contagem_paradas_por_maquina.iloc[0]
            maquina_menos_paradas = contagem_paradas_por_maquina.index[-1]
            cont_menos_paradas = contagem_paradas_por_maquina.iloc[-1]

    df_status_maquinas = pd.DataFrame({
        'Status': ['Rodando', 'Em Falha', 'Paradas'],
        'Porcentagem': [perc_rodando, perc_falha, perc_parada]
    })
    df_status_maquinas = df_status_maquinas[df_status_maquinas['Porcentagem'] > 0.001] # Evitar floats muito pequenos

    df_idle_failure = df[df['machine_status'].isin(['Idle', 'Failure'])]
    if not df_idle_failure.empty:
        contagem_status_por_falha = df_idle_failure.groupby(['failure_type', 'machine_status']).size().reset_index(name='contagem')
        contagem_status_por_falha.loc[
            (contagem_status_por_falha['machine_status'] == 'Idle') & (contagem_status_por_falha['failure_type'] == 'No Failure'),
            'failure_type'
        ] = 'Parada Programada/Outra'
        contagem_status_por_falha.loc[
            (contagem_status_por_falha['machine_status'] == 'Failure') & (contagem_status_por_falha['failure_type'] == 'No Failure'),
            'failure_type'
        ] = 'Falha (Tipo Não Especificado)'
    else:
        contagem_status_por_falha = pd.DataFrame(columns=['failure_type', 'machine_status', 'contagem'])

    df_falhas_reais = df[df['machine_status'] == 'Failure']
    df_falhas_reais = df_falhas_reais[df_falhas_reais['failure_type'] != 'No Failure']
    if not df_falhas_reais.empty:
        contagem_tipos_falha_por_maquina = df_falhas_reais.groupby(['machine', 'failure_type']).size().reset_index(name='quantidade_falhas')
    else:
        contagem_tipos_falha_por_maquina = pd.DataFrame(columns=['machine', 'failure_type', 'quantidade_falhas'])

    # Renomear colunas de valor para evitar conflitos e para clareza no melt
    media_consumo_energia = media_consumo_energia.rename(columns={'energy_consumption': 'Consumo de Energia'})
    mediatemp = mediatemp.rename(columns={'temperature': 'Temperatura'})
    media_press = media_press.rename(columns={'pressure': 'Pressão'})
    media_vibration = media_vibration.rename(columns={'vibration': 'Vibração'})

    # Unir os DataFrames
    df_medias_combinadas = mediatemp.merge(media_press, on='machine', how='left')
    df_medias_combinadas = df_medias_combinadas.merge(media_vibration, on='machine', how='left')
    df_medias_combinadas = df_medias_combinadas.merge(media_consumo_energia, on='machine', how='left')
    df_medias_combinadas.dropna(inplace=True) # Remove linhas se alguma métrica estiver faltando para uma máquina

    if not df_medias_combinadas.empty:
        df_medias_melted = df_medias_combinadas.melt(id_vars=['machine'],
                                                     value_vars=['Temperatura', 'Pressão', 'Vibração', 'Consumo de Energia'],
                                                     var_name='Metrica',
                                                     value_name='Valor Médio')
    else:
        df_medias_melted = pd.DataFrame(columns=['machine', 'Metrica', 'Valor Médio'])

    # Dados para correlação
    numeric_df = df.select_dtypes(include=np.number)
    # Remover colunas que não fazem sentido para correlação direta ou que são identificadores
    cols_to_drop_corr = ['Unnamed: 0'] # Adicione outras colunas se necessário (ex: colunas de ID se não forem numéricas por natureza)
    numeric_df = numeric_df.drop(columns=cols_to_drop_corr, errors='ignore')
    
    if numeric_df.shape[1] > 1: # Precisa de pelo menos 2 colunas numéricas para calcular correlação
        corr_pearson = numeric_df.corr(method='pearson')
        corr_spearman = numeric_df.corr(method='spearman')

    # else: corr_pearson e corr_spearman permanecem como DataFrames vazios inicializados antes do if not df.empty

else: # Caso df esteja vazio após filtros
    # Criar DataFrames vazios para evitar erros nos gráficos
    df_medias_melted = pd.DataFrame(columns=['machine', 'Metrica', 'Valor Médio'])
    df_manutencao_required = pd.DataFrame(columns=['machine', 'contagem_manutencao'])
    df_status_maquinas = pd.DataFrame(columns=['Status', 'Porcentagem'])
    contagem_status_por_falha = pd.DataFrame(columns=['failure_type', 'machine_status', 'contagem'])
    contagem_tipos_falha_por_maquina = pd.DataFrame(columns=['machine', 'failure_type', 'quantidade_falhas'])
    # corr_pearson e corr_spearman já foram inicializados como DataFrames vazios

# --------------------
#       LAYOUT STREAMLIT
# --------------------
st.title("Dashboard de Manufatura Inteligente")

# Criar abas
tab_principal, tab_correlacao, tab_tabela_dados = st.tabs(["Visão Geral", "Análise de Correlação", "Tabela de Dados"])

with tab_principal:
    if df.empty and not ('Todas' in maquinas_selecionadas or not maquinas_selecionadas): # Adicionado para cobrir caso de filtro de data resultar em df vazio
        st.warning("Nenhuma máquina selecionada ou dados disponíveis para a seleção e período.")
    elif df.empty:
        st.warning("Não há dados disponíveis para o período selecionado.")
    else:
        # Métricas de Paradas
        st.subheader("Métricas de Paradas por Máquina")
        col_metrica1, col_metrica2 = st.columns(2)
        with col_metrica1:
            st.metric(label="Máquina com Mais Paradas", value=maquina_mais_paradas, delta=f"{cont_mais_paradas} paradas", delta_color="inverse")
        with col_metrica2:
            st.metric(label="Máquina com Menos Paradas", value=maquina_menos_paradas, delta=f"{cont_menos_paradas} paradas", delta_color="normal")
        
        st.markdown("---") # Linha divisória

        col1, col2 = st.columns(2)

        with col1:
            st.header("Métricas Gerais e Manutenção")
            if not df_medias_melted.empty:
                fig_medias_combinadas_linha = px.line(df_medias_melted,
                                                      x='machine',
                                                      y='Valor Médio',
                                                      color='Metrica',
                                                      title='Médias de Sensores por Máquina',
                                                      markers=True,
                                                      labels={'machine': 'Máquina', 'Valor Médio': 'Valor Médio da Métrica', 'Metrica': 'Tipo de Métrica'})
                st.plotly_chart(fig_medias_combinadas_linha, use_container_width=True)
            else:
                st.info("Sem dados de médias de sensores para exibir para a seleção atual.")

            if not df_status_maquinas.empty:
                fig_status_maquinas_rosca = px.pie(df_status_maquinas,
                                                   values='Porcentagem',
                                                   names='Status',
                                                   title='Distribuição do Status das Máquinas (%)',
                                                   hole=0.4,
                                                   color_discrete_map={'Rodando':'green',
                                                                       'Em Falha':'red',
                                                                       'Paradas': 'orange',
                                                                       'Outros':'lightgrey'})
                st.plotly_chart(fig_status_maquinas_rosca, use_container_width=True)
            else:
                st.info("Sem dados de status das máquinas para exibir para a seleção atual.")

        with col2:
            st.header("Análise de Falhas e Paradas")
            if not contagem_status_por_falha.empty:
                fig_status_por_falha_barras = px.bar(contagem_status_por_falha,
                                                     x='failure_type',
                                                     y='contagem',
                                                     color='machine_status',
                                                     barmode='group',
                                                     title="Ocorrências de 'Parada' e 'Falha' por Causa Associada",
                                                     labels={'failure_type': "Causa Associada",
                                                             'contagem': "Número de Ocorrências",
                                                             'machine_status': "Status da Máquina"},
                                                     color_discrete_map={'Idle': 'orange',
                                                                         'Failure': 'red'})
                st.plotly_chart(fig_status_por_falha_barras, use_container_width=True)
            else:
                st.info("Sem dados de ocorrências de parada/falha para exibir para a seleção atual.")

            if not df_manutencao_required.empty:
                fig_manutencao_barras = px.bar(df_manutencao_required,
                                               x='machine',
                                               y='contagem_manutencao',
                                               title='Número de Manutenções Necessárias por Máquina',
                                               labels={'machine': 'Máquina', 'contagem_manutencao': 'Número de Manutenções'},
                                               color='machine')
                st.plotly_chart(fig_manutencao_barras, use_container_width=True)
            else:
                st.info("Sem dados de manutenção para exibir para a seleção atual.")

        # Gráfico ocupando a largura total abaixo das colunas
        st.header("Detalhes de Falhas por Máquina")
        if not contagem_tipos_falha_por_maquina.empty:
            fig_tipos_falha_maquina = px.line(contagem_tipos_falha_por_maquina,
                                                x='machine',
                                                y='quantidade_falhas',
                                                color='failure_type',
                                                markers=True,
                                                title='Quantidade de Falhas por Tipo e Máquina',
                                                labels={'machine': 'Máquina',
                                                        'quantidade_falhas': 'Número de Falhas',
                                                        'failure_type': 'Tipo de Falha'})
            st.plotly_chart(fig_tipos_falha_maquina, use_container_width=True)
        else:
            st.info("Sem dados de tipos de falha por máquina para exibir para a seleção atual.")

with tab_correlacao:
    st.header("Análise de Correlação") # Título geral da aba
    
    if df.empty:
        st.info("Não há dados para calcular a correlação com os filtros atuais.")
    # Verifica se numeric_df foi definido (ou seja, df não estava vazio) e se tem colunas suficientes
    elif 'numeric_df' not in locals() or numeric_df.shape[1] <= 1 : 
        st.info("Não há variáveis numéricas suficientes (pelo menos 2) para calcular uma matriz de correlação significativa com os filtros atuais.")
    else:
        st.subheader("Heatmap de Correlação de Pearson")
        if not corr_pearson.empty: 
            fig_corr_pearson = px.imshow(corr_pearson,
                                         text_auto=True, # Mostra os valores de correlação
                                         aspect="auto",
                                         color_continuous_scale='RdBu_r', # Escala de cores (Vermelho-Azul)
                                         title="Matriz de Correlação de Pearson entre Variáveis Numéricas")
            fig_corr_pearson.update_layout(height=700) # Ajustar altura se necessário
            st.plotly_chart(fig_corr_pearson, use_container_width=True)
            st.markdown("""
            **Correlação de Pearson:** Mede a relação linear entre duas variáveis contínuas.
            - **Valores próximos de +1:** Indicam uma forte correlação linear positiva (quando uma variável aumenta, a outra tende a aumentar).
            - **Valores próximos de -1:** Indicam uma forte correlação linear negativa (quando uma variável aumenta, a outra tende a diminuir).
            - **Valores próximos de 0:** Indicam pouca ou nenhuma correlação linear.
            """)
            # Análise da maior correlação de Pearson
            if not corr_pearson.empty and corr_pearson.shape[0] > 1:
                # Para encontrar a maior correlação, ignoramos a diagonal (correlação de uma variável consigo mesma)
                corr_pearson_no_diag = corr_pearson.mask(np.equal(*np.indices(corr_pearson.shape)))
                max_corr_pearson_val = corr_pearson_no_diag.abs().max().max()
                # Encontrar os pares
                s = corr_pearson_no_diag.abs().unstack()
                so = s.sort_values(kind="quicksort", ascending=False)
                if not so.empty:
                    pair_max_corr_pearson = so.index[0]
                    st.markdown(f"**Análise (Pearson):** O par com maior correlação linear absoluta é **{pair_max_corr_pearson[0]}** e **{pair_max_corr_pearson[1]}** com um valor de **{max_corr_pearson_val:.2f}**.")
        else: # Adicionado para o caso de corr_pearson ser vazio mesmo que numeric_df não seja (improvável com a lógica atual, mas seguro)
            st.info("Não foi possível calcular a correlação de Pearson.")

        st.subheader("Heatmap de Correlação de Spearman")
        if not corr_spearman.empty:
            fig_corr_spearman = px.imshow(corr_spearman,
                                         text_auto=True, # Mostra os valores de correlação
                                         aspect="auto",
                                         color_continuous_scale='RdBu_r', # Escala de cores (Vermelho-Azul)
                                         title="Matriz de Correlação de Spearman entre Variáveis Numéricas")
            fig_corr_spearman.update_layout(height=700) # Ajustar altura se necessário
            st.plotly_chart(fig_corr_spearman, use_container_width=True)
            st.markdown("""
            **Correlação de Spearman:** Mede a relação monotônica entre duas variáveis (sejam elas contínuas ou ordinais).
            Verifica se, à medida que uma variável aumenta, a outra tende a aumentar ou diminuir, não necessariamente a uma taxa constante (diferente da linearidade de Pearson).
            - **Valores próximos de +1:** Indicam uma forte correlação monotônica positiva.
            - **Valores próximos de -1:** Indicam uma forte correlação monotônica negativa.
            - **Valores próximos de 0:** Indicam pouca ou nenhuma correlação monotônica.
            """)
            # Análise da maior correlação de Spearman
            if not corr_spearman.empty and corr_spearman.shape[0] > 1:
                corr_spearman_no_diag = corr_spearman.mask(np.equal(*np.indices(corr_spearman.shape)))
                max_corr_spearman_val = corr_spearman_no_diag.abs().max().max()
                s_spearman = corr_spearman_no_diag.abs().unstack()
                so_spearman = s_spearman.sort_values(kind="quicksort", ascending=False)
                if not so_spearman.empty:
                    pair_max_corr_spearman = so_spearman.index[0]
                    st.markdown(f"**Análise (Spearman):** O par com maior correlação monotônica absoluta é **{pair_max_corr_spearman[0]}** e **{pair_max_corr_spearman[1]}** com um valor de **{max_corr_spearman_val:.2f}**.")
        else:
            st.info("Não foi possível calcular a correlação de Spearman.")

with tab_tabela_dados:
    st.header("Tabela de Dados Filtrados")

    # Função para converter DataFrame para CSV
    @st.cache_data # Cache para evitar reprocessamento desnecessário
    def convert_df_to_csv(input_df):
        return input_df.to_csv(index=False).encode('utf-8')

    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="Download da Tabela como CSV",
            data=csv_data,
            file_name='dados_filtrados.csv',
            mime='text/csv',
        )
    else:
        st.info("Não há dados para exibir com os filtros atuais.")
