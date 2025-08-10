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
from componentes import dataset, drop_graficos, dropdown_modelos, botao, tabs_modelos, tabs_ML, cor_fundo, cor_componentes, cor_texto 

git = '/assets/yay.png'
x_esc_inteiros = joblib.load('x_esc_inteiros.pkl') # Dados escalonados
path = 'Credit Risk Benchmark Dataset.csv'
dados_risk = pd.read_csv(path, encoding = 'utf-8')
x = dados_risk.iloc[:, :-1]
y = dados_risk.iloc[:, -1] #Target
x_treino, x_teste, y_treino, y_teste = train_test_split(x_esc_inteiros, y, random_state = 1, test_size = 0.2)
barra = px.bar(x, x='age', y='open_credit', title='Gráfico de Barra')

# Função para gerar os gráficos com base nos parâmetros selecionados
def graficar(Kernel, C, n_estimators, max_depth, modelo, tipo_grafico):
    cor_grafico = '#484545'
    cor_grade = '#1C1717'
    cor_ticks = '#6f4b4b'
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
            labels={'x': 'FPR', 'y': 'TPR'}
        )

        grafico.update_layout(
            annotations=[
                dict(
                text=legenda,
                xref="paper", yref="paper",
                x=0.01, y=1.05, showarrow=False,
                font=dict(size=14, 
                          family='Helvetica, sans-serif',
                          color = 'white'
                          ),
                )
        ],  plot_bgcolor= cor_grafico,      # fundo do quadriculado
            paper_bgcolor= cor_fundo,
            title=dict(
                text="<b>Curva ROC</b>",
                font=dict(family="Helvetica", 
                          size=16, 
                          color=cor_texto)
            ),
            xaxis=dict(
                gridcolor= cor_grade,
                  title =  'Recall', # cor das linhas verticais,
                  color = cor_ticks,
                  linecolor=cor_fundo,
                  showline = False
            ),
            yaxis=dict(
                gridcolor= cor_grade,  # cor das linhas horizontais
                title = 'Precisão',
                color = cor_ticks,
                linecolor=cor_fundo,
                showline = False         
            )
        )
        grafico.update_traces(line=dict(color=cor_ticks))   
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
                font=dict(family="Helvetica", 
                          size=14, 
                          color='white'),
                align="center"
                )
        ],  plot_bgcolor=cor_grafico,      # fundo do quadriculado
            paper_bgcolor=cor_fundo,
            title=dict(
                text="<b>Curva Precisão x Recall</b>",
                font=dict(family="Helvetica", 
                          size=16, 
                          color=cor_texto)
            ),

            xaxis=dict(
                gridcolor= cor_grade,
                  title =  'Recall',
                  color = cor_ticks,
                  linecolor=cor_fundo,
                  showline = False 
            ),
            yaxis=dict(
                gridcolor= cor_grade,  # cor das linhas horizontais
                title = 'Precisão',   
                color = cor_ticks,
                linecolor=cor_fundo,
                showline = False     
            )

                )
         
        grafico.update_traces(line=dict(color=cor_ticks)) 
        return grafico
    
kernel = 'poly'
C = 1,
n_estimators = 10
max_depth = 3
modelo = 'Random Forest'
tipo_grafico = 'Precision Curve'   
exemplo = graficar(Kernel=kernel, C=C, n_estimators=n_estimators, 
                    max_depth=max_depth, modelo=modelo, tipo_grafico=tipo_grafico)

imagem = html.Img(
        src=git,
        style={"width": "100px", "height": "auto"} 
        )
# Layout da aplicação
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB, 
                                           dbc.icons.FONT_AWESOME])
server = app.server
app.title = 'Credit Risk Benchmark Dataset'
app.layout = dbc.Container(
    [dbc.Row([
        dbc.Col(
            [
        html.H1('Credit Risk Benchmark Dataset', 
                                    style = {'backgroundColor': cor_fundo,
                                             'textAlign': 'left',
                                             'marginBottom': '15px',
                                             'marginTop': '15px',
                                             'marginLeft': '15px',
                                             'font-weight': 'bold',
                                             'font-family': 'Helvetica, sans-serif',
                                             'color': cor_texto,
                                             'font-size': '30px',
                                             }),
        html.H2('Machine Learning para Análise de Risco de Crédito',
                                    style = {'textAlign': 'left',
                                             'font-weight': 'light',
                                             'font-family': 'Helvetica, sans-serif',
                                             'font-size': '18px',
                                             'marginBottom': '25px',
                                             'marginLeft': '15px',
                                             'color': cor_texto
                                             }
        )
        ],
        width = 6
    ),
    dbc.Col([imagem], width = 3,
            style = {'textAlign': 'center',
                     'marginTop': '15px',
                     'padding': '0px'
                    }   
            ),

    dbc.Col([html.H2('GuilhermeMaceno'
                    '/Dashboard',
                    style = 
                    {
                    'font-family': 'Helvetica',
                    'fontSize': '16px',
                    'color': "#ebe6e6",
                    'textAlign': 'left',
                    'marginTop': '55px',
                    'marginLeft': '-35%',
                    'marginRight': '3%',
                    "whiteSpace": "normal",  # permite quebra automática
                    "overflowWrap": "break-word",  # quebra palavras longas
                    "wordWrap": "break-word", 
                    }
                    )],
                    width = 3, xs = 3, xl = 3, md =12
                    )
     ]
    ),

        dbc.Row(
            [html.H2('Parâmetros dos Modelos', style={'textAlign': 'left', 
                                                  'marginBottom': '5px',
                                                  'font-weight': 'regular',
                                                  'font-family': 'Helvetica, sans-serif',
                                                  'font-size': '16px',
                                                  'padding': '3px',
                                                  'marginLeft': '2%',
                                                  'color': cor_texto
                                                  }),
                dbc.Col(
                    [
                        tabs_ML,
                        html.H2('Modelos e Gráficos', 
                                style={'textAlign': 'left', 
                                        'marginBottom': '5px',
                                        'marginTop' : '15px',
                                        'font-family': 'Helvetica, sans-serif',
                                        'font-size': '16px',
                                        'color': cor_texto
                                        }),
                        dropdown_modelos,
                        drop_graficos,
                        botao,
                    ],
                    width = 3, xs = 12, sm=12, md=12, lg=3, xl=3,
                    style = {'marginBottom': '40px',
                             'border' : '3px solid #212121',
                             'borderRadius': '3px solid #212121'
                             }

                ),
                dbc.Col(
                    [
                        html.Div([
                            dcc.Graph(
                                id='grafico1', 
                                figure= exemplo,
                                style={'height': '60vh', 'width': '100%', 
                                       "border": '3px solid #212121',
                                       "borderRadius": "4px",
                                       }                         
                            )
                        ])
                    ],
                    width = 9, xl = 9, xs = 12, sm = 12, md = 12, lg =12
                        )
            ]
        ),
        dbc.Row([
    dbc.Col(
        tabs_modelos,
        xs=12, sm=12, md=12, lg=3, xl=4,  
        style={'padding': '0px', 'margin': '0px'}
    ),
    dbc.Col(
        dataset,
        xs=12, sm=12, md=12, lg=9, xl=7,
        style={'textAlign': 'left', 'marginBottom': '20px'}
    )
])
            ],
        fluid=True,
        style={"backgroundColor": cor_fundo}
)
    
           

@app.callback(
    Output(component_id = 'grafico1', component_property = 'figure'),
    Input(component_id = 'botao_graficar', component_property = 'n_clicks'),
    State(component_id = 'menu1', component_property = 'value'),
    State(component_id = 'SVM kernel parâmetros', component_property = 'value'),
    State(component_id = 'SVM_C_parâmetros', component_property = 'value'),
    State(component_id = 'Rdn F n_estimators parâmetros', component_property = 'value'),
    State(component_id = 'Profundidade Máxima', component_property = 'value'),
    State(component_id = 'menu_modelos', component_property = 'value')
)
def att_grafico(n_clicks, tipo_grafico, kernel, C, n_estimators, max_depth, modelo):
    if n_clicks == 0:
        return exemplo
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
