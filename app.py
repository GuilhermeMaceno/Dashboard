from dash import Dash, html, dcc # type: ignore
from dash.dependencies import Input, Output, State # type: ignore
import dash_bootstrap_components as dbc # type: ignore
import plotly.express as px # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.ensemble import BaggingClassifier # type: ignore
from componentes import dataset, drop_graficos, dataset_col, markdown_modelos, botao, tabs_modelos, tabs_ML, tab_svm, tab_random_forest   

exemplo = joblib.load('salvo.pkl') # O gráfico inicial
x_esc_inteiros = joblib.load('x_esc_inteiros.pkl') # Dados escalonados
path = 'Credit Risk Benchmark Dataset.csv'
dados_risk = pd.read_csv(path, encoding = 'utf-8')
x = dados_risk.iloc[:, :-1]
y = dados_risk.iloc[:, -1] #Target
x_treino, x_teste, y_treino, y_teste = train_test_split(x_esc_inteiros, y, random_state = 1, test_size = 0.2)
barra = px.bar(x, x='age', y='open_credit', title='Gráfico de Barra')

# Função para gerar os gráficos com base nos parâmetros selecionados
def graficar(Kernel, C, n_estimators, max_depth, modelo, tipo_grafico):
    legenda = f"Modelo: {modelo} | Kernel: {Kernel} | C: {C} | n_estimators: {n_estimators} | max_depth: {max_depth}"
    if modelo == 'SVM':
        classificador = SVC(C=C, kernel=Kernel)
        classificador.fit(x_treino, y_treino)

    if modelo == 'Random Forest':
        classificador = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators, random_state=42)
        classificador.fit(x_treino, y_treino)

    if tipo_grafico == 'ROC Curve':
        fpr, tpr, _ = roc_curve(y_teste, classificador.predict(x_teste))
        grafico = px.line(
            x=fpr, y=tpr, 
            title='Curva ROC', 
            labels={'x': 'FPR', 'y': 'TPR'},
        )

        grafico.update_layout(
            annotations=[
                dict(
                text=legenda,
                xref="paper", yref="paper",
                x=0.01, y=1.05, showarrow=False,
                font=dict(size=15, family='Helvetica, sans-serif'),
                align="center"
                )
        ],  plot_bgcolor="#EBF5FF",      # fundo do quadriculado
            paper_bgcolor="#ffffff",
            title=dict(
                text="<b>Curva ROC</b>",
                font=dict(family="Helvetica", 
                          size=20, 
                          color="black")
            ),
            xaxis_title="FPR",
            yaxis_title="TPR",
)
        return grafico
    
    if tipo_grafico == 'Precision Curve':
        precisions, recalls, _ = precision_recall_curve(y_teste, classificador.predict(x_teste))
        grafico = px.line(
            x=recalls, 
            y=precisions, 
            title='Curva de Precisão', 
            labels={'x': 'Recall', 'y': 'Precision'}
        )
        grafico.update_layout(
            annotations=[
                dict(
                text=legenda,
                xref="paper", yref="paper",
                x=0.01, y=1.05, showarrow=False,
                font=dict(size=16, family='Helvetica, sans-serif'),
                align="center"
                )
        ],  plot_bgcolor="#EBF5FF",      # fundo do quadriculado
            paper_bgcolor="#ffffff",
            title=dict(
                text="<b>Curva Precisão x Recall</b>",
                font=dict(family="Helvetica", 
                          size=20, 
                          color="black")
            ),
            xaxis_title="Recall",
            yaxis_title="Precisão",
                )
            
        
        return grafico

# Layout da aplicação
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB, dbc.icons.FONT_AWESOME])
server = app.server
app.title = 'Credit Risk Benchmark Dataset'
app.layout = dbc.Container(
    [
        html.H1('Credit Risk Benchmark Dataset', 
                                    style = {'backgroundColor': 'white',
                                             'textAlign': 'center',
                                             'padding': '2px',
                                             'font-weight': 'bold',
                                             'font-family': 'Helvetica, sans-serif',
                                             'marginTop': '30px',
                                             }),
        html.H2('Machine Learning para Análise de Risco de Crédito',
                                    style = {'textAlign': 'center',
                                             'font-weight': 'light',
                                             'font-family': 'Helvetica, sans-serif',
                                             'font-size': '20px',
                                             'padding': '0.1px',
                                             }
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div([
                            dcc.Graph(
                                id='grafico1', 
                                figure= exemplo,
                                style={'height': '60vh', 'width': '100%'}                         
                            )
                        ])
                    ],
                    width = 9
                        ),
                dbc.Col(
                    [
                        tabs_ML,
                        botao,
                        dataset_col
                    ],
                    width = 3

                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [   html.H2('Modelos e Gráficos', 
                                style={'textAlign': 'left', 
                                        'marginBottom': '20px',
                                        'font-weight': 'bold',
                                        'font-family': 'Helvetica, sans-serif',
                                        'font-size': '20px',
                                        'padding': '3px',
                                        'marginLeft': '2px'}),
                        markdown_modelos,
                        drop_graficos,
                        tabs_modelos
                    ],
                    width = 9,
                    style={'marginLeft': '30px'}
                ),
            ]
        )
            ],
    fluid=True,
        )
    
           

@app.callback(
    Output(component_id = 'grafico1', component_property = 'figure'),
    Input(component_id = 'botao_graficar', component_property = 'n_clicks'),
    State(component_id = 'menu1', component_property = 'value'),
    State(component_id = 'SVM kernel parâmetros', component_property = 'value'),
    State(component_id = 'SVM C parâmetros', component_property = 'value'),
    State(component_id = 'Rdn F n_estimators parâmetros', component_property = 'value'),
    State(component_id = 'Profundidade Máxima', component_property = 'value'),
    State(component_id = 'menu_modelos', component_property = 'value')
)
def att_grafico(n_clicks, tipo_grafico, kernel, C, n_estimators, max_depth, modelo):
    if C == None:
        C = 1
        
    if n_estimators == None:
        n_estimators = 10

    if max_depth == None:
        max_depth = 3
    
    n_estimators = int(n_estimators)
    max_depth = int(max_depth) 

    
    return graficar(Kernel=kernel, C=C, n_estimators=n_estimators, 
                    max_depth=max_depth, modelo=modelo, tipo_grafico=tipo_grafico)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
