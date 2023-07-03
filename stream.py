import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta
import base64
import textwrap
from PIL import Image

# def render_svg(svg):
#     """Renders the given svg string."""
#     b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
#     html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
#     st.write(html, unsafe_allow_html=True)

def render_svg_example():
    image = Image.open('./Transparent Logo.png')
    col1, col2, col3 = st.columns(3)
    with col1:
        pass
    with col2:
        st.image(image, width=400)
    with col3:
        pass




df = pd.read_csv('sanciones_utf.csv', sep=';')
df.columns = df.columns.str.replace('de', 'De').str.replace(' ', '').str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df['FechaDeImposicion'] = df['FechaDeImposicion'].apply(lambda x: datetime(1900, 1, 1)+ timedelta(days=x - 2))
df['FechaDeImposicion'] = pd.to_datetime(df['FechaDeImposicion'])
df['Monto'] = df['Monto'].str.replace('$', '').str.replace(',', '').astype(float)
#columns Subsector, TipoDeSancion into a lower case and delete multiple spaces or tabs or new lines
df['Subsector'] = df['Subsector'].str.lower().str.replace('\s+', ' ', regex=True)
df['TipoDeSancion'] = df['TipoDeSancion'].str.lower().str.replace('\s+', ' ', regex=True)

st.set_page_config(
    page_title='Multas Dashboard', 
    page_icon=':bar_chart:', 
    layout='wide'
)

render_svg_example()

st.subheader('Multas Dashboard')

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
df_multas = df.query('TipoDeSancion == "multa (sanción pecuniaria)"').groupby('Subsector')['Monto'].sum().to_frame().sort_values(by='Monto', ascending=False).reset_index()
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
        df['Subsector'].unique(),
        index = 6
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

# #top infractores por monto acumulado o por numero de multas
# df_multas_acumuladas = df.Subsector.to_list()
# df_multas_acumuladas.append('FechaDeImposicion')
# df_multas_acumuladas.append('Monto')
# df_multas_acumuladas = df.query('Subsector in @df_multas_acumuladas').groupby(['Subsector', 'FechaDeImposicion'])['Monto'].sum().to_frame().reset_index()
# df_multas_acumuladas['TotalAcumulado'] = df_multas_acumuladas.groupby('Subsector')['Monto'].cumsum()
# df_multas_acumuladas['TotalAcumulado'] = df_multas_acumuladas['TotalAcumulado'].astype(int)
# df_multas_acumuladas['TotalAcumulado'] = df_multas_acumuladas['TotalAcumulado'].astype(str)
# df_multas_acumuladas['TotalAcumulado'] = df_multas_acumuladas['TotalAcumulado'].apply(lambda x: f'${x}')
# left_column, right_column = st.columns(2)
# with left_column:
#     st.subheader('Top infractores por monto acumulado')
#     top_n = st.slider('Select top n', 3, df_multas_acumuladas.shape[0], 3)
# with right_column:
#     pass
# left_column, right_column = st.columns(2)
# with left_column:
#     st.dataframe(df_multas_acumuladas.groupby('Subsector')['TotalAcumulado'].max().sort_values(ascending=False).head(top_n))
# with right_column:
#     fig = px.bar(df_multas_acumuladas.groupby('Subsector')['TotalAcumulado'].max().sort_values(ascending=False).head(top_n), x='Subsector', y='TotalAcumulado', color='TotalAcumulado', color_continuous_scale=px.colors.sequential.Viridis)
#     st.plotly_chart(fig, use_container_width=True)





st.subheader('Full Dataset')
st.dataframe(df)