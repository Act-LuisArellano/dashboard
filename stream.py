import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import base64
import textwrap
from PIL import Image
import numpy as np
import re

def remove_extra_spaces(text):
    # Remove multiple spaces before and after a phrase
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove multiple spaces between words
    text = re.sub(r'\s+', ' ', text)

    return text

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
df['Subsector'] = df['Subsector'].str.lower().str.strip().str.replace('\s+', ' ', regex=True)
df['TipoDeSancion'] = df['TipoDeSancion'].str.lower().str.strip().str.replace('\s+', ' ', regex=True)

st.set_page_config(
    page_title='Multas Dashboard', 
    page_icon=':bar_chart:', 
    layout='wide'
)

render_svg_example()

st.title('Multas Dashboard')

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
top_n = st.slider('Select top n', 3, df_multas.shape[0], 3, key='multas')
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

#top infractores por monto acumulado
st.title('Infractores')
df_infractores = df.groupby('Infractor')['Monto'].sum().to_frame().sort_values(by='Monto', ascending=False).reset_index()
st.subheader('Top infractores por monto acumulado')
top_n_infractores = st.slider('Select top n', 3, df_infractores.shape[0], 3, key='infractores')
st.metric(label="Total Acumulado", value=f'${round(df_infractores.head(top_n_infractores).Monto.sum()/1000000,0)}M', delta=f'${round(df_infractores.head(top_n_infractores).iloc[-1].Monto/1000000,0)}M')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_infractores.head(top_n_infractores))
with right_column:
    fig = px.bar(df_infractores.head(top_n_infractores), x='Infractor', y='Monto', color='Monto', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

#top infractores por numero de multas
st.subheader('Top infractores por numero de multas')
df_infractores = df.groupby('Infractor')['Monto'].count().to_frame().sort_values(by='Monto', ascending=False).reset_index()
top_n_ixm = st.slider('Select top n', 3, df_infractores.shape[0], 3, key='infractores_ixm')
st.metric(label="Total Acumulado", value=f'{round(df_infractores.head(top_n_ixm).Monto.sum(),0)}', delta=f'{round(df_infractores.head(top_n_ixm).iloc[-1].Monto,0)}')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_infractores.head(top_n_ixm))
with right_column:
    fig = px.bar(df_infractores.head(top_n_ixm), x='Infractor', y='Monto', color='Monto', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

#top conducta sancionada (col ConductaSancionada)
st.title('Conducta Sancionada')
df_conducta_sancionada = df.groupby('ConductaSancionada')['Monto'].sum().to_frame().sort_values(by='Monto', ascending=False).reset_index()
st.subheader('Top conducta sancionada por monto acumulado')
top_n_conducta_sancionada = st.slider('Select top n', 3, df_conducta_sancionada.shape[0], 3, key='conducta_sancionada')
st.metric(label="Total Acumulado", value=f'${round(df_conducta_sancionada.head(top_n_conducta_sancionada).Monto.sum()/1000000,0)}M', delta=f'${round(df_conducta_sancionada.head(top_n_conducta_sancionada).iloc[-1].Monto/1000000,0)}M')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_conducta_sancionada.head(top_n_conducta_sancionada))
with right_column:
    fig = px.bar(df_conducta_sancionada.head(top_n_conducta_sancionada), x='ConductaSancionada', y='Monto', color='Monto', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

#top conducta sancionada (col ConductaSancionada) por numero de multas
st.subheader('Top conducta sancionada por numero de multas')
df_conducta_sancionada = df.groupby('ConductaSancionada')['Monto'].count().to_frame().sort_values(by='Monto', ascending=False).reset_index()
top_n_conducta_sancionada = st.slider('Select top n', 3, df_conducta_sancionada.shape[0], 3, key='conducta_sancionada_ixm')
st.metric(label="Total Acumulado", value=f'{round(df_conducta_sancionada.head(top_n_conducta_sancionada).Monto.sum(),0)}', delta=f'{round(df_conducta_sancionada.head(top_n_conducta_sancionada).iloc[-1].Monto,0)}')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_conducta_sancionada.head(top_n_conducta_sancionada))
with right_column:
    fig = px.bar(df_conducta_sancionada.head(top_n_conducta_sancionada), x='ConductaSancionada', y='Monto', color='Monto', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)

#top conducta sancionada por monto acumulado (add mean and stddev col), mean and stddev format to 2 decimals (add numero de multas col)
df_conducta_sancionada = df.groupby('ConductaSancionada')['Monto'].agg(['sum', 'median' , 'mean', 'std', 'count']).sort_values(by='sum', ascending=False).reset_index()
df_conducta_sancionada['sum'] = df_conducta_sancionada['sum'].apply(lambda x: round(x, 2))
df_conducta_sancionada['mean'] = df_conducta_sancionada['mean'].apply(lambda x: round(x, 2))
df_conducta_sancionada['median'] = df_conducta_sancionada['median'].apply(lambda x: round(x, 2))
df_conducta_sancionada['std'] = df_conducta_sancionada['std'].apply(lambda x: round(x, 2))
st.subheader('Top conducta sancionada de monto acumulado con mediana, media y desviación estándar')
top_n_conducta_sancionada = st.slider('Select top n', 3, df_conducta_sancionada.shape[0], 3, key='conducta_sancionada_mas_ixm')
st.metric(label="Total Acumulado", value=f'${round(df_conducta_sancionada.head(top_n_conducta_sancionada)["sum"].sum()/1000000,0)}M', delta=f'${round(df_conducta_sancionada.head(top_n_conducta_sancionada)["sum"].iloc[-1]/1000000,0)}M')
left_column, right_column = st.columns(2)
with left_column:
    st.dataframe(df_conducta_sancionada.head(top_n_conducta_sancionada))
with right_column:
    fig = px.bar(df_conducta_sancionada.head(top_n_conducta_sancionada), x='ConductaSancionada', y='median', color='median', color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)




#sales experiment
st.title('Máxima ganancia posible')
#add streamlit multiselect of Subsector col values for exclude in dataframe
subsector_excluded = st.multiselect(
    'Selecciona el subsector a excluir',
    df['Subsector'].unique(),
    default=[np.nan]
)
#price range streamlit inputs
st.subheader('Rango de precios')
left_column, right_column = st.columns(2)
with left_column:
    min_price = st.number_input('Precio mínimo anual', value=100000, step=1000)
with right_column:
    max_price = st.number_input('Precio máximo anual', value=300000, step=1000)

#query df with excluded values
df_q = df.query('Subsector not in @subsector_excluded')
prices = np.arange(min_price, max_price, 1000)
df_sorted = df_q.sort_values(by='Monto')
sales = [(df_sorted['Monto'] > price).sum() for price in prices]
data = {'price': prices, 'sales': sales}
df_plot = pd.DataFrame(data)
df_plot['profit'] = df_plot['price'] * df_plot['sales']
# Define the min and max color values
min_color = df_plot['profit'].min()
max_color = df_plot['profit'].max()
# Define the desired midpoint of the color scale
mid_color = (max_color + min_color) / 2
# Find the price that maximizes profit
max_profit_index = df_plot['profit'].idxmax()
max_profit_price = df_plot.loc[max_profit_index, 'price']


fig = px.scatter(df_plot, x='price', y='sales', color='profit',
                 color_continuous_scale='Viridis',
                 range_color=[min_color, max_color],
                 color_continuous_midpoint=mid_color)

# Update color bar's length and thickness
fig.update_layout(coloraxis_colorbar=dict(len=0.65, thickness=10))
# Add a vertical line at the price that maximizes profit
fig.add_trace(go.Scatter(x=[max_profit_price, max_profit_price], y=[df_plot['sales'].min(), df_plot['sales'].max()], mode='lines', name='Max Profit Price', line=dict(color='red')))
st.plotly_chart(fig, use_container_width=True)
#indicator of max profit price, sales needed to achieve max profit price, and optimal price
left, center, right = st.columns(3)
with left:
    st.metric(label="Precio anual óptimo", value=f'${round(max_profit_price,0)}', delta =f'${round(max_profit_price/12,0)} mensuales' )
with center:
    st.metric(label="Ventas", value=f'{round(df_plot.loc[max_profit_index, "sales"],0)}', delta=f'{round((df["FechaDeImposicion"].max() - df["FechaDeImposicion"].min()).days / 365,2)} Años')
with right:
    st.metric(label="Ganancia máxima", value=f'${round(df_plot.loc[max_profit_index, "profit"]/1_000_000,2)}M')

st.markdown("***")

st.title('Full Dataset')
st.dataframe(df)