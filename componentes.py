import dash_bootstrap_components as dbc #type: ignore
from dash import dcc, html #type: ignore

dataset = dbc.Card([
    dbc.CardHeader(html.H3('Sobre o Dataset'),
                   style={
                       'font-family': 'Helvetica, Arial, sans-serif',
                       'font-weight': 'bold',
                       'background-color': "#FFFFFF",
                       'font-size': '14px',
                         }
                   ),
    dcc.Markdown('''
**Overview**:
                The data comprises a 
                 mix of financial metrics and personal attributes, which allow 
                 users to build and evaluate models for credit risk scoring.
                 Link: https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset
                 
                 ''',
                 style = {
                     #'padding': '5px',
                     'margin-top': '10px',
                         }
    )
                ],
    body=True, 
    className='mt-3',
    id = 'dataset_card',
    style={
        'border': '1px solid #dee2e6',
        'border-radius': '0.25rem',
        'background-color': "#ffffff",
        })


drop_graficos = dcc.Dropdown(id = 'menu1',
            options=
            [
            {'label': 'ROC Curve', 'value': 'ROC Curve'},
            {'label': 'Precision Curve', 'value': 'Precision Curve'}
            ],
            value='ROC Curve',
            style={'width': '95%', 'marginTop': '1px'})

dataset_col = dbc.Collapse(
            dataset,
            id="collapse_dataset",
            is_open=True,
            style={"position": "absolute", 
                   "zIndex": 999, "width": "20%",
                   'marginTop': '30px',
                   }
        )

markdown_modelos = dcc.Dropdown(
    id='menu_modelos',
    options=[
        {'label': 'SVM', 'value': 'SVM'},
        {'label': 'Random Forest', 'value': 'Random Forest'},
        ],
        value='SVM',
        style={'width': '95%', 'marginTop': '1px'}
)

botao = dbc.Button('Ok', 
                   color = 'primary', 
                   className = "me-md-2",
                   n_clicks = 0,
                   id = 'botao_graficar',
                   style = {'marginTop': '20px', 'width': '30%',
                            'color': 'light'
                            },
                   outline = True,
                   size = 'small',
                   )

tabs_modelos = html.Div([
    dbc.Tabs([
        dbc.Tab(
            [dcc.Markdown(
                '''
                
                Parâmetros do SVM:
                - **Kernel**: linear, polynomial, radial basis function (RBF)
                - **C**: Regularization parameter
                - **Gamma**: Kernel coefficient for RBF
                - **Degree**: Degree of the polynomial kernel function (if applicable)
                ''', 
                id = 'Markdown_svm')
            ],
            label = 'SVM', 
            tab_id = 'tab1'
                ),

        dbc.Tab(
            [dcc.Markdown('''
                          
                Parâmetros do Random Forest:
                
                - **n_estimators**: Number of trees in the forest
                - **max_depth**: Maximum depth of the tree''',
                id = 'markdown_random_forest')
            ],
            tab_id = 'tab2',
            label = 'Random Forest'
                )
            ],
        id = 'tabs_modelos',
        active_tab = 'tab1')
    ])

# Menu de parâmetros dos modelos
tab_svm = dbc.Tab(
    [
    html.Label('Tipo de Kernel', style = {'marginTop': '20px'}),
    dcc.Dropdown(
        id = 'SVM kernel parâmetros',
        options=
            [
            {'label': 'Kernel Radial', 'value': 'rbf'},
            {'label': 'Kernel Linear', 'value': 'linear'},
            {'label': 'Kernel Polynomial', 'value': 'Poly'},      
            ],
            value='linear',
            placeholder='Select SVM Parameter'
                ),

    html.Label('Parâmetro C', style = {'marginTop': '20px'}),
    dbc.InputGroup(
        [
            dbc.Input(
                id='SVM C parâmetros',
                type='number',
                min=0.1,
                max=100,
                placeholder='Enter C value',)
        ]
        )
    ],
    tab_id='tab_suporte_vector_machine',
    label = 'SVM'
        )

tab_random_forest = dbc.Tab(
    [
        html.Label('Número de Estimadores', style = {'marginTop': '20px'}),
        dbc.InputGroup(
        [
            dbc.Input(
                value = 100,
                id='Rdn F n_estimators parâmetros',
                type='integer',
                min=1,
                max=1000,
                placeholder='Estimators value',)
        ]
        ),
        html.Label('Profundidade Máxima', style = {'marginTop': '20px'}),
        dbc.InputGroup(
        [
            dbc.Input(
                value = 10,
                id='Profundidade Máxima',
                type='integer',
                min=1,
                max=1000,
                placeholder='Max depth value' 
                )
        ]
        )
    ],
    tab_id='menu_parametros_rf',
    label = 'Rdn F'
)

tabs_ML = html.Div(
    [
    html.H2('Parâmetros dos Modelos', style={'textAlign': 'left', 
                                                  'marginBottom': '20px',
                                                  'font-weight': 'bold',
                                                  'font-family': 'Helvetica, sans-serif',
                                                  'font-size': '20px',
                                                  'padding': '3px',
                                                  'marginLeft': '2px'}),
dbc.Tabs(
    [
        tab_svm,
        tab_random_forest
    ],
    id='tabs_modelos_ML',
    active_tab='tab_suporte_vector_machine',
    style={
        'marginTop': '20px',
        'marginLeft': '10px',
        'width': '100%',
        'font-weight': 'bold',
        'font-family': 'Helvetica, Arial, sans-serif',
        'background-color': "#FFFFFF",
        'color': '#000000',
        'font-size': '14px'}
)
    ],
    id = 'div_ML'
)