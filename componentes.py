import dash_bootstrap_components as dbc #type: ignore
from dash import dcc, html #type: ignore

cor_fundo = "#1C1717"
cor_componentes = "#212121"
cor_texto = "#746F6F"

dataset = dbc.Card([
    dbc.CardHeader(html.H3('Sobre o Dataset',
                        style=
                        {
                       'font-family': 'Helvetica, Arial, sans-serif',
                       'font-weight': 'bold',
                       'font-size': '18px',
                       'color': cor_texto,
                       'textAlign': 'center'                         
                        }
                           ),
                   
                   ),
    dcc.Markdown('''
**Overview**:
                The data comprises a 
                 mix of financial metrics and personal attributes, which allow 
                 users to build and evaluate models for credit risk scoring.
                 Link: https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset
                 
                 ''',
                 style = {
                     'margin-top': '10px',
                     'fontSize': '14px'
                         }
    )
                ],
    body=True, 
    className='mt-3',
    id = 'dataset_card',
    style={
        'border': '1px solid #dee2e6',
        'border-radius': '0.25rem',
        'background-color': cor_componentes,
        'color': cor_texto
        })


drop_graficos = dcc.Dropdown(id = 'menu1',
            options=
            [
            {'label': 'ROC Curve', 'value': 'ROC Curve'},
            {'label': 'Precision Curve', 'value': 'Precision Curve'}
            ],
            value='ROC Curve',
            style={'width': '100%', 
                   'marginTop': '1px',
                   'background-color': cor_componentes,
                   'fontSize': '13px'
                   }
                )

dataset_col = dbc.Collapse(
            dataset,
            id="collapse_dataset",
            is_open=True,
            style={"position": "absolute", 
                   "zIndex": 999, "width": "20%",
                   'marginTop': '30px',
                   'background-color': cor_componentes
                   }
        )

dropdown_modelos = dcc.Dropdown(
    id='menu_modelos',
    options=[
        {'label': 'SVM', 'value': 'SVM'},
        {'label': 'Random Forest', 'value': 'Random Forest'},
        ],
        value='SVM',
        style={'width': '100%', 
               'marginTop': '1px',
               'background-color': cor_componentes,
               'border' : '0px solid white',
               'outline': 'none',
               'fontSize': '13px'
               }
)

botao = dbc.Button('Ok', 
                   color = cor_componentes,
                   className = "me-md-2",
                   n_clicks = 0,
                   id = 'botao_graficar',
                   style = {'marginTop': '60px', 'width': '18%',
                            'color': 'white',
                            'backgroundColor': "#624848",
                            'fontSize': '13px',
                            'marginBottom': '35px' 
                            },
                   outline = False,
                   size = 'small',
                   )

tabs_modelos = html.Div(
    dbc.Tabs(
        [
            dbc.Tab(
                dcc.Markdown(
                    '''
                    Parâmetros do SVM:
                    - **Kernel**: linear, polynomial, radial basis function (RBF)
                    - **C**: Regularization parameter
                    - **Gamma**: Kernel coefficient for RBF
                    - **Degree**: Degree of the polynomial kernel function (if applicable)
                    ''',
                    id='Markdown_svm',
                    style={"marginTop": "10px", "padding": "0px"}  # remove espaço interno
                ),
                label='SVM',
                style={'color': cor_texto, 'margin': '0px', 'fontSize': '14px'},
                label_style={'color': cor_componentes},  # aba mais estreita
                tab_id='tab1'
            ),

            dbc.Tab(
                dcc.Markdown(
                    '''
                    Parâmetros do Random Forest:
                    - **n_estimators**: Number of trees in the forest
                    - **max_depth**: Maximum depth of the tree
                    ''',
                    id='markdown_random_forest',
                    style={"marginTop": "10px", "padding": "0px"}
                ),
                label='Random Forest',
                style={'color': cor_texto, 'margin': '0px', 'fontSize': '14px'},
                label_style={'color': cor_componentes},
                tab_id='tab2'
            )
        ],
        id='tabs_modelos',
        active_tab='tab1',
        style={'border': '0px', 'padding': '0px', 'margin': '0px'}  # borda e espaços zerados
    ),
    style={'padding': '0px', 
           'marginLeft': '15px',
           'marginTop': '5px'}
)


# Menu de parâmetros dos modelos
tab_svm = dbc.Tab(
    [
    html.Label('Tipo de Kernel', 
               style = {'marginTop': '20px',
                        'color': cor_texto
                        }),
    dcc.Dropdown(
        id = 'SVM kernel parâmetros',
        options=
            [
            {'label': 'Kernel Radial', 'value': 'rbf'},
            {'label': 'Kernel Linear', 'value': 'linear'},
            {'label': 'Kernel Polynomial', 'value': 'poly'},      
            ],
            value='linear',
            placeholder='Select SVM Parameter',
            style = {'backgroundColor': cor_componentes,
                    'color': cor_componentes,
                    'borderRadius': '5px',
                    'border' : '0px solid #6D6B6B',
                    'fontSize': '13px'
                    } 
                ),

    html.Label('Parâmetro C', 
               style = {'marginTop': '20px',
                        'color': cor_texto
                        }),
    dbc.InputGroup(
        [
            dbc.Input(
                value = 1,
                id='SVM_C_parâmetros',
                type='number',
                min=0.1,
                max=100,
                placeholder='_______',
                style = {'backgroundColor': cor_componentes,
                         'color': cor_texto,
                         'borderRadius': '5px',
                         'border' : '0px solid #6D6B6B',
                         'fontSize': '13px'
                         }, 
                className = 'Inputs'
                        )
        ],
        style = {'backgroundColor': cor_componentes,
                'color': cor_texto,
                'border': '2px',  
                'borderRadius': '5px'} 
        )
    ],
    tab_id='tab_suporte_vector_machine',
    label_style = {'color': cor_componentes},
    label = 'SVM'
        )

tab_random_forest = dbc.Tab(
    [
        html.Label('Número de Estimadores', 
                   style = {'marginTop': '20px',
                            'color': cor_texto,
                            'backgroundColor': cor_fundo
                            }),
        dbc.InputGroup(
        [
            dbc.Input(
                value = 10,
                id='Rdn F n_estimators parâmetros',
                type='integer',
                min=1,
                max=1000,
                placeholder='Estimators value',
                style = {'backgroundColor': cor_componentes,
                         'color': cor_texto,
                         'borderRadius': '5px',
                         'border' : '0px solid #6D6B6B',
                         'fontSize': '13px'
                         },
                className = 'Inputs'
                )
                
        ]
        ),
        html.Label('Profundidade Máxima', 
                   style = {'marginTop': '20px',
                            'color': cor_texto,
                            'backgroundColor': cor_fundo
                            }),
        dbc.InputGroup(
        [
            dbc.Input(
                value = 3,
                id='Profundidade Máxima',
                type='integer',
                min=1,
                max=1000,
                placeholder='Max depth value',
                style = {'backgroundColor': cor_componentes,
                         'color': cor_texto,
                         'borderRadius': '5px',
                         'border' : '0px solid #6D6B6B',
                         'fontSize': '13px'
                         },
                className = 'Inputs' 
                )
        ]
        )
    ],
    tab_id='menu_parametros_rf',
    label_style = {'color': cor_componentes},
    label = 'Rdn F'
)

tabs_ML = html.Div(
    [
dbc.Tabs(
    [
        tab_svm,
        tab_random_forest
    ],
    id='tabs_modelos_ML',
    active_tab='tab_suporte_vector_machine',
    style={
        'marginTop': '5px',
        'marginLeft': '10px',
        'width': '100%',
        'font-weight': 'bold',
        'font-family': 'Helvetica, Arial, sans-serif',
        'background-color': cor_fundo,
        'color': '#000000',
        'font-size': '14px',
        'border' : '0px solid white'
        }
)
    ],
    id = 'div_ML'
)