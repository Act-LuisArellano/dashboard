import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta


df = pd.read_csv('sanciones_utf.csv', sep=';')
df.columns = df.columns.str.replace('de', 'De').str.replace(' ', '').str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df['FechaDeImposicion'] = df['FechaDeImposicion'].apply(lambda x: datetime(1900, 1, 1)+ timedelta(days=x - 2))
df['FechaDeImposicion'] = pd.to_datetime(df['FechaDeImposicion'])
df['Monto'] = df['Monto'].str.replace('$', '').str.replace(',', '').astype(float)

st.set_page_config(
    page_title='Multas Dashboard', 
    page_icon=':bar_chart:', 
    layout='wide'
)

st.title('Multas Dashboard')
st.subheader('Dataset')
st.dataframe(df)

st.subheader('Date range')
st.warning('Selected date range would be applied to all the charts')
left_column, right_column = st.columns(2)
with left_column:
    start_date = st.date_input('Start date', value=datetime(2018,12,4))
with right_column:
    end_date = st.date_input('End date', value=df['FechaDeImposicion'].max())
#filter df by date range
df = df.query('FechaDeImposicion >= @start_date and FechaDeImposicion <= @end_date')

st.subheader('Tipo de Sanción')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df.groupby('TipoDeSancion')['TipoDeSancion'].count().to_frame())
with right_column:
    fig = px.pie(df, names='TipoDeSancion', color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

st.title('Multas')
df_multas = df.query('TipoDeSancion == "Multa (Sanción Pecuniaria)"').groupby('Subsector')['Monto'].sum().to_frame().sort_values(by='Monto', ascending=False).reset_index()
st.subheader('Montos totales por subsector')
top_n = st.slider('Select top n', 3, df_multas.shape[0], 3)
st.metric(label="Total Acumulado", value=f'${round(df_multas.head(top_n).Monto.sum()/1000000,0)}M', delta=f'${round(df_multas.head(top_n).iloc[-1].Monto/1000000,0)}M')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_multas.head(top_n))
with right_column:
    fig = px.bar(df_multas.head(top_n), x='Subsector', y='Monto', color='Monto', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

st.subheader('Series de tiempo para montos por subsector')
df_multas_top_n = df_multas.head(top_n).Subsector.to_list()
df_multas_top_n.append('FechaDeImposicion')
df_multas_top_n.append('Monto')
df_multas_top_n = df.query('Subsector in @df_multas_top_n').groupby(['Subsector', 'FechaDeImposicion'])['Monto'].sum().to_frame().reset_index()
fig = px.line(df_multas_top_n, x='FechaDeImposicion', y='Monto', color='Subsector', color_discrete_sequence=px.colors.sequential.Viridis)
st.plotly_chart(fig, use_container_width=True)



df_multas_acumuladas = df.Subsector.to_list()
df_multas_acumuladas.append('FechaDeImposicion')
df_multas_acumuladas.append('Monto')
df_multas_acumuladas = df.query('Subsector in @df_multas_acumuladas').groupby(['Subsector', 'FechaDeImposicion'])['Monto'].sum().to_frame().reset_index()
df_multas_acumuladas['TotalAcumulado'] = df_multas_acumuladas.groupby('Subsector')['Monto'].cumsum()
left_column, right_column = st.columns(2)
with left_column:
    st.subheader('Multas acumuladas por subsector')
    subsector = st.selectbox(
        'Select a Subsector',
        df['Subsector'].unique()
    )
with right_column:
    pass
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_multas_acumuladas.query('Subsector == @subsector').sort_values(by='FechaDeImposicion', ascending=True))
with right_column:
    fig = px.line(df_multas_acumuladas.query('Subsector == @subsector'), x='FechaDeImposicion', y='TotalAcumulado', color='Subsector', color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_traces(fill='tozeroy')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

