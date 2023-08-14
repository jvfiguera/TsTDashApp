# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import dash_bootstrap_templates as dbt

fig_bar_bystate =''
key_graph       =''
graph_colptte_list   =[] # [top_labels,x_data,y_data]
graph_tunel_list    =[]
fontsize_title_graph_glb =14
#GRAPH_PARAM_DICT = { 'key':[p_title,p_x,p_y,p_color,p_text,p_xlabel,p_ylabel] }
GRAPH_PARAM_DICT ={'id-homepage1':['VOLUMEN TRANSACCIONAL POR ESTADO','ESTADO','TOTAL_TRX','TIPO TRANSACCION','TIPO TRANSACCION','Estado','Cantidad de Transacciones']
                    ,'id-graph01-trx-bystate':['VOLUMEN TRANSACCIONAL MENSUAL POR ESTADO','ESTADO','TOTAL_TRX',None,None,'Estado','Cantidad de Transacciones']
                    ,'id-graph_pie01-trx-bystate':['APROBADAS vs RECHAZADAS','STATUS_TRX','TOTAL_TRX','None','None','None','None']
                    ,'id-graph-line_bydate':['VOLUMEN TRANSACCIONAL MENSUAL POR DIA','FECHA_TRX','TOTAL_TRX','STATUS_TRX','None','None','None']
                    ,'id-graph01-bar-term':['PARQUE DE EQUIPOS POR ESTADO','ESTADO','CANTIDAD_EQP','TIPO_COMUNICACION','TIPO_COMUNICACION','Estados','Cantidad de Equipos']
                    ,'id-terminals-park' :['SUNBURST - DISTRIBUCIÓN PARQUE EQUIPOS',['ADQUIRIENCIA','TIPO_TERMINAL','TIPO_COMUNICACION'],'CANTIDAD_EQP','CANTIDAD_EQP',['TIPO_TERMINAL','TIPO_COMUNICACION'],'CANTIDAD_EQP','CANTIDAD_EQP']
                    ,'id-graph01-donuts-entadq': ['EQUIPOS SEGÚN ADQUIRIENCIA','ADQUIRIENCIA','CANTIDAD_EQP',None,None,None,None]
                    ,'id-graph01-bar-tunnel': ['ADQUIRIENCIA vs.TIPO DE COMUNICACIÓN',None,None,None,None,None,None]
                    #,'id-graph01-tipo-term': ['EQUIPOS SEGÚN TIPO DE TERMINAL','TIPO_TERMINAL','CANTIDAD_EQP',None,None,None,None]
                    ,'id-graph01-tipo-term': ['EQUIPOS SEGÚN TIPO DE TERMINAL','ADQUIRIENCIA','CANTIDAD_EQP','TIPO_TERMINAL','TIPO_TERMINAL','Adquiriencia','Cantidad de Equipos']
                   }

# Starting Functions Definitions

def fn_read_file(pfile_name,psep):
    ''' Función para la lectura de archivos'''
    return pd.read_csv(filepath_or_buffer=pfile_name,sep=psep)

def fn_group_data_bystate(p_data_set,p_sep,p_groupby_list,p_sumby):
    ''' Función para la agrupración de la data del Data Files '''
    df_data             = fn_read_file(pfile_name=p_data_set, psep=p_sep)
    #df_data[p_groupby_list[2]] = df_data[p_groupby_list[2]].str.capitalize()
    df_data             = pd.DataFrame(df_data.groupby(p_groupby_list)[p_sumby].sum())
    df_data             = df_data.reset_index()
    return df_data

def fn_groupby_dataset(p_data_set,p_groupby_list,p_sumby):
    ''' Función para la agrupración de la data del Data Files '''
    df_data             = pd.DataFrame(p_data_set.groupby(p_groupby_list)[p_sumby].sum())
    df_data             = df_data.reset_index()
    return df_data

def fn_filter_dataset(p_data_set,p_strfilter):
    #p_strfilter="FECHA_TRX >= '01/06/2023' and FECHA_TRX<= '30/06/2023' and ESTADO=='Aragua'"
    df_datset_fitered = p_data_set.query(p_strfilter)
    return df_datset_fitered

# def fn_prepare_dataset_page1(p_df_data_parkterm):
#     global graph_colptte_list
#     graph_colptte_list.clear()
#     df_data_state_tcom      = fn_groupby_dataset(p_data_set=p_df_data_parkterm, p_groupby_list=['ESTADO','TIPO_COMUNICACION'],p_sumby='CANTIDAD_EQP')
#     df_grapth_state_tcom    =df_data_state_tcom
#     df_data_state_tcom['%'] =(df_data_state_tcom['CANTIDAD_EQP'] / df_data_state_tcom['CANTIDAD_EQP'].sum()) * 100
#     res_data_dict = {'ESTADO': 'TOTAL', 'CANTIDAD_EQP': df_data_state_tcom['CANTIDAD_EQP'].sum(), '%': df_data_state_tcom['%'].sum()}
#     df_data_state_tcom = df_data_state_tcom.to_dict(orient='records')
#     df_data_state_tcom.append(res_data_dict)
#     df_data_state_tcom = pd.DataFrame.from_dict(df_data_state_tcom)
#     df_data_state_tcom['CANTIDAD_EQP'] = df_data_state_tcom["CANTIDAD_EQP"].map('{:,d}'.format)
#     df_data_state_tcom['%'] = df_data_state_tcom["%"].map('{:.2f}'.format)
#
#     df_sunburst_datset = fn_groupby_dataset(p_data_set=p_df_data_parkterm, p_groupby_list=['ADQUIRIENCIA','ESTADO','TIPO_TERMINAL','TIPO_COMUNICACION','STATUS_TERMINAL'], p_sumby='CANTIDAD_EQP')
#     df_donut_entadq    = fn_groupby_dataset(p_data_set=p_df_data_parkterm,p_groupby_list=['ADQUIRIENCIA'],p_sumby='CANTIDAD_EQP')
#     df_gbar_colptte    = fn_groupby_dataset(p_data_set=p_df_data_parkterm,p_groupby_list=['ADQUIRIENCIA','TIPO_COMUNICACION'],p_sumby='CANTIDAD_EQP')
#     top_labels_list    = df_gbar_colptte['TIPO_COMUNICACION'].sort_values().unique().tolist()
#     graph_colptte_list.append(top_labels_list)
#     y_data              =df_gbar_colptte['ADQUIRIENCIA'].sort_values().unique().tolist()
#     graph_colptte_list.append(y_data)
#     tot_sum =df_gbar_colptte['CANTIDAD_EQP'].astype(int).sum()
#     x_data_list =df_gbar_colptte.to_dict(orient="split")
#     x_data      =[]
#     edp_dialup  =0
#     eqp_inalam  =0
#     eqp_lan     =0
#     eqp_tcpip   =0
#
#     # MERCANTIL
#     for i in range (len(x_data_list['data'])):
#         if x_data_list['data'][i][0] =='BANCO MERCANTIL':
#             if x_data_list['data'][i][1] == 'DialUp':
#                  edp_dialup = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'Inalambrico':
#                  eqp_inalam = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'LAN':
#                  eqp_lan = (x_data_list['data'][i][2]/tot_sum)*100
#             else: #TCP-IP
#                  eqp_tcpip = (x_data_list['data'][i][2]/tot_sum)*100
#     x_data.append([np.round(edp_dialup, 2), np.round(eqp_inalam, 2), np.round(eqp_lan, 2), np.round(eqp_tcpip, 2)])
#
#     edp_dialup  = 0
#     eqp_inalam  = 0
#     eqp_lan     = 0
#     eqp_tcpip   = 0
#     # PROVINCIAL
#     for i in range(len(x_data_list['data'])):
#         if x_data_list['data'][i][0] == 'BANCO PROVINCIAL':
#             if x_data_list['data'][i][1] == 'DialUp':
#                 edp_dialup = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'Inalambrico':
#                 eqp_inalam = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'LAN':
#                 eqp_lan = (x_data_list['data'][i][2]/tot_sum)*100
#             else:  # TCP-IP
#                 eqp_tcpip = (x_data_list['data'][i][2]/tot_sum)*100
#     x_data.append([np.round(edp_dialup,2),np.round(eqp_inalam,2), np.round(eqp_lan,2),np.round(eqp_tcpip,2)])
#
#     edp_dialup  = 0
#     eqp_inalam  = 0
#     eqp_lan     = 0
#     eqp_tcpip   = 0
#     # COMÚN
#     for i in range(len(x_data_list['data'])):
#         if x_data_list['data'][i][0] == 'COMUN':
#             if x_data_list['data'][i][1] == 'DialUp':
#                 edp_dialup = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'Inalambrico':
#                 eqp_inalam = (x_data_list['data'][i][2]/tot_sum)*100
#             elif x_data_list['data'][i][1] == 'LAN':
#                 eqp_lan = (x_data_list['data'][i][2]/tot_sum)*100
#             else:  # TCP-IP
#                 eqp_tcpip = (x_data_list['data'][i][2]/tot_sum)*100
#     x_data.append([np.round(edp_dialup, 2), np.round(eqp_inalam, 2), np.round(eqp_lan, 2), np.round(eqp_tcpip, 2)])
#
#     graph_colptte_list.append(x_data)
#
#     return df_grapth_state_tcom,df_data_state_tcom,df_sunburst_datset,df_donut_entadq,df_gbar_colptte,graph_colptte_list

def fn_prepare_dataset_page1(p_df_data_parkterm,p_query_filter):
    global graph_tunel_list
    graph_tunel_list.clear()
    if p_query_filter!=None:
        df_data_filter = fn_filter_dataset(p_data_set=p_df_data_parkterm, p_strfilter=p_query_filter)
    else:
        df_data_filter=p_df_data_parkterm

    df_data_state_tcom      = fn_groupby_dataset(p_data_set=df_data_filter, p_groupby_list=['ESTADO','TIPO_COMUNICACION'],p_sumby='CANTIDAD_EQP')
    df_grapth_state_tcom    =df_data_state_tcom
    df_data_state_tcom['%'] =(df_data_state_tcom['CANTIDAD_EQP'] / df_data_state_tcom['CANTIDAD_EQP'].sum()) * 100
    res_data_dict = {'ESTADO': 'TOTAL', 'CANTIDAD_EQP': df_data_state_tcom['CANTIDAD_EQP'].sum(), '%': df_data_state_tcom['%'].sum()}
    df_data_state_tcom = df_data_state_tcom.to_dict(orient='records')
    df_data_state_tcom.append(res_data_dict)
    df_data_state_tcom = pd.DataFrame.from_dict(df_data_state_tcom)
    df_data_state_tcom['CANTIDAD_EQP'] = df_data_state_tcom["CANTIDAD_EQP"].map('{:,d}'.format)
    df_data_state_tcom['%'] = df_data_state_tcom["%"].map('{:.2f}'.format)

    df_sunburst_datset = fn_groupby_dataset(p_data_set=df_data_filter, p_groupby_list=['ADQUIRIENCIA','ESTADO','TIPO_TERMINAL','TIPO_COMUNICACION','STATUS_TERMINAL'], p_sumby='CANTIDAD_EQP')
    df_donut_entadq    = fn_groupby_dataset(p_data_set=df_data_filter,p_groupby_list=['ADQUIRIENCIA'],p_sumby='CANTIDAD_EQP')
    df_gbar_colptte    = fn_groupby_dataset(p_data_set=df_data_filter,p_groupby_list=['ADQUIRIENCIA','TIPO_COMUNICACION'],p_sumby='CANTIDAD_EQP')
    df_pie_tipterm     = fn_groupby_dataset(p_data_set=df_data_filter,p_groupby_list=['ADQUIRIENCIA', 'TIPO_TERMINAL'], p_sumby='CANTIDAD_EQP')
    top_labels_list    = df_gbar_colptte['TIPO_COMUNICACION'].sort_values().unique().tolist()
    graph_tunel_list.append(top_labels_list)
    y_data              =df_gbar_colptte['ADQUIRIENCIA'].sort_values().unique().tolist()
    graph_tunel_list.append(y_data)
    #tot_sum =df_gbar_colptte['CANTIDAD_EQP'].astype(int).sum()
    x_data_list =df_gbar_colptte.to_dict(orient="split")
    x_data      =[]
    edp_dialup  =0
    eqp_inalam  =0
    eqp_lan     =0
    eqp_tcpip   =0

    # MERCANTIL
    i='BANCO MERCANTIL'
    if i in y_data:
        for i in range (len(x_data_list['data'])):
            if x_data_list['data'][i][0] =='BANCO MERCANTIL':
                if x_data_list['data'][i][1] == 'DialUp':
                     edp_dialup = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'Inalambrico':
                     eqp_inalam = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'LAN':
                     eqp_lan = x_data_list['data'][i][2]
                else: #TCP-IP
                     eqp_tcpip = x_data_list['data'][i][2]

        x_data.append([edp_dialup, eqp_inalam,eqp_lan,eqp_tcpip])

    edp_dialup  = 0
    eqp_inalam  = 0
    eqp_lan     = 0
    eqp_tcpip   = 0
    # PROVINCIAL
    i='BANCO PROVINCIAL'
    if i in y_data:
        for i in range(len(x_data_list['data'])):
            if x_data_list['data'][i][0] == 'BANCO PROVINCIAL':
                if x_data_list['data'][i][1] == 'DialUp':
                    edp_dialup = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'Inalambrico':
                    eqp_inalam = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'LAN':
                    eqp_lan = x_data_list['data'][i][2]
                else:  # TCP-IP
                    eqp_tcpip = x_data_list['data'][i][2]

        x_data.append([edp_dialup,eqp_inalam, eqp_lan,eqp_tcpip])

    edp_dialup  = 0
    eqp_inalam  = 0
    eqp_lan     = 0
    eqp_tcpip   = 0
    # COMÚN
    i = 'COMUN'
    if i in y_data:
        for i in range(len(x_data_list['data'])):
            if x_data_list['data'][i][0] == i:
                if x_data_list['data'][i][1] == 'DialUp':
                    edp_dialup = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'Inalambrico':
                    eqp_inalam = x_data_list['data'][i][2]
                elif x_data_list['data'][i][1] == 'LAN':
                    eqp_lan = x_data_list['data'][i][2]
                else:  # TCP-IP
                    eqp_tcpip = x_data_list['data'][i][2]

        x_data.append([edp_dialup,eqp_inalam, eqp_lan,eqp_tcpip])

    graph_tunel_list.append(x_data)

    return df_grapth_state_tcom,df_data_state_tcom,df_sunburst_datset,df_donut_entadq,df_gbar_colptte,graph_tunel_list,df_pie_tipterm

def fn_prepare_dataset_page2(p_df_data_filter,p_query_filter,p_filt,p_start_date,p_bank,p_state):
    # Grupo de Graficos de la parte superior del dashborad
    metrics_dict = {}
    if p_filt =='G':
        if p_bank== 'All Banks' and p_state =='All States':
            # Filtrar solo por año#
            glb_query_filter     = f"YEAR == {int(pd.to_datetime(p_start_date).strftime('%Y'))}"
        elif p_bank!='All Banks' and p_state=='All States':
            glb_query_filter = f"YEAR == {int(pd.to_datetime(p_start_date).strftime('%Y'))} and ENT_ADQ == '{p_bank}'"
        elif p_bank=='All Banks' and p_state!='All States':
            glb_query_filter = f"YEAR == {int(pd.to_datetime(p_start_date).strftime('%Y'))} and ESTADO == '{p_state}'"
        else:
            glb_query_filter = f"YEAR == {int(pd.to_datetime(p_start_date).strftime('%Y'))} and ENT_ADQ == '{p_bank}' and ESTADO=='{p_state}'"

        #Obtener el promedio global del Periodo
        df_data_filter = fn_filter_dataset(p_data_set=p_df_data_filter, p_strfilter=glb_query_filter)
        avg_year = df_data_filter['TOTAL_TRX'].mean()    #Promedio Global de referencia
        avg_year = pd.to_numeric('{:.2f}'.format(avg_year))

        #Obtener la Mediana Global del periodo
        mediana_year = df_data_filter['TOTAL_TRX'].median()    #Mediana Global de referencia
        mediana_year = pd.to_numeric('{:.2f}'.format(mediana_year))

        # Obtener el promedio del periodo
        df_data_filter  = fn_filter_dataset(p_data_set=p_df_data_filter, p_strfilter=p_query_filter)
        df_data_period   =df_data_filter
        df_data_period  = fn_groupby_dataset(p_data_set=df_data_period, p_groupby_list=['FECHA_TRX','STATUS_TRX'], p_sumby='TOTAL_TRX')
        avg_period      = df_data_filter['TOTAL_TRX'].mean() # Promedio del periodo de evaluación
        avg_period      = pd.to_numeric('{:.2f}'.format(avg_period))
        mediana_period  = df_data_filter['TOTAL_TRX'].median() # Mediana del periodo de evaluación
        mediana_period  = pd.to_numeric('{:.2f}'.format(mediana_period))

        # Preparando las metricas
        metrics_dict['avg']     =(avg_year,avg_period)
        metrics_dict['mediana'] =(mediana_year,mediana_period)
        metrics_dict['mediana'] = (mediana_year, mediana_period)

        # procesar
        df_tbl_data         = fn_groupby_dataset(p_data_set=df_data_filter, p_groupby_list=['ESTADO'], p_sumby='TOTAL_TRX')
        df_tbl_data         = pd.DataFrame(df_tbl_data[['ESTADO', 'TOTAL_TRX']])
        df_tbl_data['%']    = (df_tbl_data['TOTAL_TRX'] / df_tbl_data['TOTAL_TRX'].sum()) * 100
        res_data_dict       = {'ESTADO': 'TOTAL', 'TOTAL_TRX': df_tbl_data['TOTAL_TRX'].sum(), '%': df_tbl_data['%'].sum()}
        df_tbl_final        = df_tbl_data.to_dict(orient='records')
        df_tbl_final.append(res_data_dict)
        df_tbl_final        = pd.DataFrame.from_dict(df_tbl_final)
        df_tbl_final['TOTAL_TRX'] = df_tbl_final["TOTAL_TRX"].map('{:,d}'.format)
        df_tbl_final['%']   = df_tbl_final["%"].map('{:.2f}'.format)

        # Grupo de Graficos de la parte inferior del dashborad
        df_graph_pie01 = fn_groupby_dataset(p_data_set=df_data_filter, p_groupby_list=['STATUS_TRX'], p_sumby='TOTAL_TRX')
        df_barchart_colpltte = fn_groupby_dataset(p_data_set=df_data_filter, p_groupby_list=['TIPO_TRX'],p_sumby='TOTAL_TRX')
        df_barchart_colpltte['%'] = (df_barchart_colpltte['TOTAL_TRX'] / df_barchart_colpltte['TOTAL_TRX'].sum()) * 100

        res_colpltte_dict = {'TIPO_TRX': 'TOTAL', 'TOTAL_TRX': df_barchart_colpltte['TOTAL_TRX'].sum(), '%': df_barchart_colpltte['%'].sum()}
        df_barchart_colpltte = df_barchart_colpltte.to_dict(orient='records')
        df_barchart_colpltte.append(res_colpltte_dict)
        df_barchart_colpltte = pd.DataFrame.from_dict(df_barchart_colpltte)
        df_barchart_colpltte['TOTAL_TRX'] = df_barchart_colpltte["TOTAL_TRX"].map('{:,d}'.format)
        df_barchart_colpltte['%'] = df_barchart_colpltte["%"].map('{:.2f}'.format)


    return df_tbl_data,df_tbl_final,df_graph_pie01,df_barchart_colpltte,metrics_dict,df_data_period



# Ending Functions Definitions

# df_data_bystate = fn_group_data_bystate(p_data_set='cvs-files/REP-RES-TRX-JUNIO2023.csv'
#                                          ,p_sep=';'
#                                          ,p_groupby_list=['YEAR', 'FECHA_TRX','ENT_ADQ', 'ESTADO','STATUS_TRX','TIPO_TRX']
#                                          ,p_sumby='TOTAL_TRX'
#                                          )
df_data_bystate  = fn_read_file(pfile_name='cvs-files/REP-RES-TRX-JUNIO2023.csv', psep=';')
df_data_parkterm = fn_read_file(pfile_name='cvs-files/REP-CANTIDAD-POS-BYSTATE.csv', psep=';')
df_data_parkterm.sort_values(by=['ESTADO','TIPO_COMUNICACION'], inplace=True)

banks_pag1     =df_data_parkterm['ADQUIRIENCIA'].sort_values().unique()
tip_term_pag1  =df_data_parkterm['TIPO_TERMINAL'].sort_values().unique()
banks_pag1     =np.append([banks_pag1],['All Banks'])
tip_term_pag1  =np.append([tip_term_pag1],['All Terminals'])

# Filtros de pagina Nro.2
states =df_data_bystate["ESTADO"].sort_values().unique()
banks  =df_data_bystate["ENT_ADQ"].sort_values().unique()
states =np.append([states],['All States'])
banks  =np.append([banks],['All Banks'])

external_stylesheets = [dbc.themes.CYBORG, dbc.icons.FONT_AWESOME,"https://fonts.googleapis.com/css2?family=Courgette&family=Foldit&family=Kavoon&family=Lilita+One&family=Metal+Mania&family=Montserrat&family=PT+Serif+Caption&family=Ultra&display=swap"]

#app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME], meta_tags=[ {"name":"viewport","content":"width=device-width", "initial-scale":"1", "shrink-to-fit":"no"}], suppress_callback_exceptions=True)
app = Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=[ {"name":"viewport","content":"width=device-width", "initial-scale":"1", "shrink-to-fit":"no"}], suppress_callback_exceptions=True)


offcanvas = html.Div(
    children=[
            dbc.Row(
                children=[
                dbc.Col(
                      html.Div(
                        children=[
                            dbc.Button(html.I( className='fa-solid fa-bars') , color="primary", id="open-offcanvas", n_clicks=0, className="me-1")
                            #,html.I(className="fa-brands fa-free-code-camp")
                            ,html.Img(src=app.get_asset_url(path='images/telefonica-img.png'), alt='img-telef') #"Portal de Ciencia de Datos"
                                ]
                            ,className='c-main-div-title'
                            )
                            ,style = {'max-width': '20%'}
                        )
                ,dbc.Col(
                      html.Div(
                        children=[
                                   html.H1('DASHBOARD DE CIENCIA DE DATOS', className='c-h1-title-dash')
                                ]
                              )
                        ,className='c-main-div2-title'
                         )
                    ]
                    ,className='c-first-row-header'
                   )
            ,dbc.Offcanvas( children=[
                                html.H1('Portal de Ciencia de Datos', className='c-H1-offcanvas-title')
                                ,html.P("Prototipo de dashboard o portal de ciencia de datos, diseñado solo con fines didacticos para mostrar la versatilidad de las librerias dash & plotly para el estudio y analísis de los datos "
                                        , style={"text-align": "justify"}
                                        )
                                ,html.Hr(id='id-line')
                                ,dbc.Nav(
                                        html.Div( children=[
                                                dbc.NavLink("Parque de Terminales", href="/id-terminals-park", active="exact")
                                                ,dbc.NavLink("Volumen Transaccional", href="/id-graph01-trx-bystate")
                                                #,dbc.NavLink("Home3", href="/id-homepage3")
                                                ]
                                                )
                                        ,vertical=True
                                        ,pills=True
                                        )

                                ]
            ,id="offcanvas"
            ,title=html.Img(src=app.get_asset_url(path='images/telefonica-img.png'), alt='img-telef') #"Portal de Ciencia de Datos"
            ,is_open=False
            ,className='text-bg-dark'
                        )
            ,dbc.Row(
                dbc.Col(
                   html.Div(id='id-page-content1',children=[])
                        )
                    )
                ,dbc.Col(children=[
                    html.Footer( children=[
                                        html.H1([' © All rights reserved. | Design by Jorge V. Figuera'] , style={"font-size": "1rem"})
                                           ]
                                )
                                ]
                        ,id='id-copyright'
                        )
            ]
            )

app.layout =html.Div( children=[dcc.Location(id='url', refresh=False)
                                ,offcanvas
                                ]
                    )
# PRIMER CALLBACK
@app.callback(
    Output(component_id="offcanvas", component_property="is_open")
    ,Output(component_id="id-page-content1", component_property="children")
    ,Input(component_id="open-offcanvas", component_property="n_clicks")
    ,Input(component_id='url',component_property='pathname')
    ,[State(component_id="offcanvas", component_property="is_open")]
    ,prevent_initial_call=True
            )
def toggle_offcanvas(n1,pathname, is_open):
    global df_data_bystate,GRAPH_PARAM_DICT,key_graph
    print('PRIMER CALLBACK:'+ pathname)
    if not pathname.replace('/', '') :
        pathname='id-terminals-park'
    key_graph = pathname.replace('/', '')
    page_layout = fn_get_page_layout(p_key_graph=key_graph)
    if n1:
        return not is_open,page_layout

    return is_open,page_layout

# SEGUNDO CALLBACK
@app.callback(
    Output('id-graph02-trx-bystate','figure')
    ,Output('id-graph_pie01-trx-bystate','figure')
    ,Output('id-dash-tbl-states', 'data')
    ,Output('id-dash-tbl-tipo-trx','data')
    ,Output('id-graph_ind_bystate','figure')
    ,Output('id-graph-line_bydate','figure')
    ,Input('url','pathname')
    ,Input('id-banks-filter','value')
    ,Input('id-states-filter','value')
    ,Input("id-date-range", "start_date")
    ,Input("id-date-range", "end_date")
    ,prevent_initial_call = True
    )
def fn_update_trx_bystate(p_pathname,p_bank,p_state,p_start_date,p_end_date):
    global key_graph,df_data_bystate, GRAPH_PARAM_DICT
    print('SEGUNDO CALLBACK:' + p_pathname)
    key_graph = p_pathname.replace('/', '')
    p_key_graph=key_graph
    p_start_date = pd.to_datetime(p_start_date).strftime("%Y-%m-%d")
    p_end_date   = pd.to_datetime(p_end_date).strftime("%Y-%m-%d")
    if p_start_date > p_end_date:
        p_end_date = p_start_date # Para evitar error la llamada al CALLBACK

    if p_key_graph == 'id-graph01-trx-bystate':
        # Definir el filtro que sera aplicado al data set
        if f"{p_bank}-{p_state}"=='All Banks-All States': # Filtro solo por fechas
            query_filter = f"FECHA_TRX >= '{p_start_date}' and FECHA_TRX<= '{p_end_date}'"
        elif p_bank !='All Banks' and p_state =='All States': # Filtro solo por fechas y banco
            query_filter =f"FECHA_TRX >= '{p_start_date}' and FECHA_TRX<= '{p_end_date}' and ENT_ADQ=='{p_bank}'"
        elif p_bank == 'All Banks' and p_state != 'All States': # Filtro solo por fechas y estado
            query_filter =f"FECHA_TRX >= '{p_start_date}' and FECHA_TRX<= '{p_end_date}' and ESTADO=='{p_state}'"
        elif p_bank != 'All Banks' and p_state != 'All States':  # Filtro solo por fechas, banco y  estado
            query_filter = f"FECHA_TRX >= '{p_start_date}' and FECHA_TRX<= '{p_end_date}' and ENT_ADQ=='{p_bank}' and ESTADO=='{p_state}'"
        else:
            query_filter = f"FECHA_TRX >= '{p_start_date}' and FECHA_TRX<= '{p_end_date}'"

        df_tbl_data,df_tbl_final,df_graph_pie01,df_barchart_colpltte,metricts_dict,df_data_period=fn_prepare_dataset_page2(p_df_data_filter=df_data_bystate,p_query_filter=query_filter,p_filt='G',p_start_date=p_start_date,p_bank=p_bank,p_state=p_state)
        fig_bar_bystate = fn_barchar_withtext(p_data_set    =df_tbl_data
                                              , p_title     =GRAPH_PARAM_DICT[p_key_graph][0]
                                              , p_x         =GRAPH_PARAM_DICT[p_key_graph][1]
                                              , p_y         =GRAPH_PARAM_DICT[p_key_graph][2]
                                              , p_color     =GRAPH_PARAM_DICT[p_key_graph][3]
                                              , p_text      =GRAPH_PARAM_DICT[p_key_graph][4]
                                              , p_xlabel    =GRAPH_PARAM_DICT[p_key_graph][5]
                                              , p_ylabel    =GRAPH_PARAM_DICT[p_key_graph][6]
                                              , p_orientacion='v'
                                              , p_color_map   =None
                                              ,p_height       =400
                                              ,p_pattern_shape=None
                                              ,p_pattern_shape_seq_list=None
                                              )
        fig_pie01_graph = fn_get_piechart(p_data_set    =df_graph_pie01
                                          ,p_title      =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][0]       # 'VOLUMEN TRANSACCIONAL APROBADAS Vs RECHAZADAS'
                                          ,p_names      =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][1]  #'STATUS_TRX'
                                          ,p_values     =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][2]  #'TOTAL_TRX'
                                          ,p_color_discrete_map={'No Definidas': 'lightcyan', 'Rechazadas': 'magenta','Aprobadas': 'cyan'}
                                          ,p_hole       =0
                                          )

        fig_indicators = fn_indicators(p_metrics_dict =metricts_dict)
        fig_line = fn_line_chart(p_data_set=df_data_period
                                 ,p_title  =GRAPH_PARAM_DICT['id-graph-line_bydate'][0]
                                 ,p_x      =GRAPH_PARAM_DICT['id-graph-line_bydate'][1] #'FECHA_TRX'
                                 ,p_y      =GRAPH_PARAM_DICT['id-graph-line_bydate'][2] #'TOTAL_TRX'
                                 ,p_color  =GRAPH_PARAM_DICT['id-graph-line_bydate'][3] #'STATUS_TRX'
                                 )

        return fig_bar_bystate,fig_pie01_graph,df_tbl_final.to_dict('records'),df_barchart_colpltte.to_dict('records'),fig_indicators,fig_line

    elif p_key_graph == 'id-terminals-park':
        page_layout = html.Div(children=[html.P('PARQUE DE TERMINALES')])
        return page_layout
    else:
        #print('RETORNANDO -------')
        page_layout = fn_get_page_layout(p_key_graph=key_graph)
        return page_layout

# TERCER CALLBACK
@app.callback(
    Output('id-dash-tbl01-term','data')
    ,Output('id-graph02-sunburst','figure')
    ,Output('id-graph01-bar-term', 'figure')
    ,Output('id-graph01-bar-tunnel','figure')
    ,Output('id-graph01-donuts-entadq','figure')
    ,Output('id-graph01-tipo-term','figure')
    ,Input('url','pathname')
    ,Input('id-banks-filter','value')
    # ,Input('id-tipterm-filter','value')
    ,prevent_initial_call = True
    )
#def fn_update_parkterm(p_pathname,p_bank,p_tipo_term):
def fn_update_parkterm(p_pathname, p_bank):
    global key_graph,df_data_parkterm, GRAPH_PARAM_DICT
    #print('TERCER CALLBACK:' + p_pathname)
    key_graph = p_pathname.replace('/', '')
    p_key_graph=key_graph
    #if p_key_graph == 'id-terminals-park':
    # Definir el filtro que sera aplicado al data set
    if f"{p_bank}"=='All Banks': # Ningún tipo de filtro
        query_filter = None
    elif p_bank !='All Banks' :# Filtro solo por banco
        query_filter =f"ADQUIRIENCIA=='{p_bank}'"
    #elif p_bank == 'All Banks' and p_tipo_term != 'All Terminals': # Filtro solo por tipo de terminal
        #query_filter =f"TIPO_TERMINAL=='{p_tipo_term}'"
    #elif p_bank != 'All Banks' and p_tipo_term != 'All Terminals':  # Filtro solo por banco y  tipo de terminal
        #query_filter = f"ADQUIRIENCIA=='{p_bank}' and TIPO_TERMINAL=='{p_tipo_term}'"

    df_grapth_state_tcom, df_data_state_tcom, df_sunburst_datset, df_donut_entadq, df_gbar_colptte, graph_tunel_list,df_pie_tipterm = fn_prepare_dataset_page1(p_df_data_parkterm=df_data_parkterm, p_query_filter=query_filter)

    fig01_bar_term = fn_barchar_withtext(p_data_set =df_grapth_state_tcom
                                         , p_title  =GRAPH_PARAM_DICT['id-graph01-bar-term'][0]
                                         , p_x      =GRAPH_PARAM_DICT['id-graph01-bar-term'][1]
                                         , p_y      =GRAPH_PARAM_DICT['id-graph01-bar-term'][2]
                                         , p_color  =GRAPH_PARAM_DICT['id-graph01-bar-term'][3]
                                         , p_text   =GRAPH_PARAM_DICT['id-graph01-bar-term'][4]
                                         , p_xlabel =GRAPH_PARAM_DICT['id-graph01-bar-term'][5]
                                         , p_ylabel =GRAPH_PARAM_DICT['id-graph01-bar-term'][6]
                                         , p_orientacion='v'
                                         , p_color_map={'DialUp': '#9D3C72', 'LAN': '#1F6E8C', 'Inalambrico': '#4E9F3D','DialUp': '#D21312'}
                                         , p_height=500
                                         ,p_pattern_shape=None
                                         ,p_pattern_shape_seq_list=None
                                         )

    fig_sunburst = fn_graph_sunburst(p_dataset  =df_sunburst_datset
                                     , p_title  =GRAPH_PARAM_DICT['id-terminals-park'][0]
                                     , p_path   =GRAPH_PARAM_DICT['id-terminals-park'][1]
                                     , p_values =GRAPH_PARAM_DICT['id-terminals-park'][2]  # 'CANTIDAD_EQP'
                                     , p_color  =GRAPH_PARAM_DICT['id-terminals-park'][3]  # 'CANTIDAD_EQP'
                                     , p_hover_data=GRAPH_PARAM_DICT['id-terminals-park'][4]
                                     , p_avg    =GRAPH_PARAM_DICT['id-terminals-park'][5]  # 'CANTIDAD_EQP'
                                     , p_weights=GRAPH_PARAM_DICT['id-terminals-park'][6]  # 'CANTIDAD_EQP'
                                     )

    fig_donut_entadq = fn_get_piechart(p_data_set   =df_donut_entadq
                                       , p_title    =GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][0]
                                       , p_names    =GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][1]
                                       , p_values   =GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][2]
                                       , p_color_discrete_map={'BANCO PROVINCIAL': 'lightcyan','BANCO MERCANTIL': '#205295', 'COMUN': '#CD1818'}
                                       , p_hole=0.4
                                       )

    # fig01_pie_tipterm = fn_get_piechart(p_data_set=df_pie_tipterm
    #                                     , p_title=GRAPH_PARAM_DICT['id-graph01-tipo-term'][0]
    #                                     , p_names=GRAPH_PARAM_DICT['id-graph01-tipo-term'][1]
    #                                     , p_values=GRAPH_PARAM_DICT['id-graph01-tipo-term'][2]
    #                                     , p_color_discrete_map={'BANCO PROVINCIAL': 'lightcyan','BANCO MERCANTIL': '#205295', 'COMUN': '#CD1818'}
    #                                     , p_hole=0
    #                                     )
    fig01_pie_tipterm = fn_barchar_withtext(p_data_set  =df_pie_tipterm
                                            , p_title   =GRAPH_PARAM_DICT['id-graph01-tipo-term'][0]
                                            , p_x       =GRAPH_PARAM_DICT['id-graph01-tipo-term'][1]
                                            , p_y       =GRAPH_PARAM_DICT['id-graph01-tipo-term'][2]
                                            , p_color   =GRAPH_PARAM_DICT['id-graph01-tipo-term'][3]
                                            , p_text    =GRAPH_PARAM_DICT['id-graph01-tipo-term'][4]
                                            , p_xlabel  =GRAPH_PARAM_DICT['id-graph01-tipo-term'][5]
                                            , p_ylabel  =GRAPH_PARAM_DICT['id-graph01-tipo-term'][6]
                                            , p_orientacion='v'
                                            , p_color_map={'CAJA REGIST.': '#9D3C72', 'POS': '#1F6E8C'}
                                            , p_height  =400
                                            , p_pattern_shape=GRAPH_PARAM_DICT['id-graph01-tipo-term'][4]
                                            , p_pattern_shape_seq_list=[".", "x"]
                                            )
    #print(graph_tunel_list)
    fig_tunnel = fn_stucked_funnel_graph(p_title    =GRAPH_PARAM_DICT['id-graph01-bar-tunnel'][0]
                                         , p_stucfunn_dat_list=graph_tunel_list)

    return df_data_state_tcom.to_dict('records'),fig_sunburst, fig01_bar_term,fig_tunnel,fig_donut_entadq,fig01_pie_tipterm

# Inicio de las funciones de Graficos
def fn_barchar_withtext(p_data_set,p_title,p_x,p_y,p_color,p_text,p_xlabel,p_ylabel,p_orientacion,p_color_map,p_height,p_pattern_shape,p_pattern_shape_seq_list):
    ''' Función para la generación de graficos de barra con texto'''
    global fontsize_title_graph_glb
    #print(p_text,p_color)
    fig_bar =px.bar(data_frame  =p_data_set
                    ,x          =p_x
                    ,y          =p_y
                    ,text_auto  = '.2s'
                    ,color      =p_color
                    ,text       =p_text
                    ,labels     ={p_x:p_xlabel, p_y:p_ylabel}
                    ,height     =p_height
                    ,orientation=p_orientacion
                    #,color_discrete_sequence = px.colors.sequential.RdBu
                    #,color_discrete_map={'DialUp': 'lightcyan', 'LAN': 'magenta', 'Inalambrico': 'cyan', 'DialUp':'red'}
                    ,color_discrete_map=p_color_map
                    ,pattern_shape=p_pattern_shape
                    ,pattern_shape_sequence=p_pattern_shape_seq_list
                    )

    fig_bar.update_layout(title=dict(
                                    text="<b>" + f'{p_title}' + "</b>"
                                    ,font=dict(size=fontsize_title_graph_glb)
                                    ,x=0.5
                                    ,xref="paper"
                                     )
                            ,title_font_color='#346751'
                            ,template='plotly_dark' # others include seaborn, ggplot2, plotly_white, plotly_dark
                           )
    return fig_bar

def fn_get_piechart(p_data_set,p_title,p_names,p_values,p_color_discrete_map,p_hole):
    global fontsize_title_graph_glb
    fig_pie = px.pie(p_data_set
                 ,values=p_data_set[p_values]
                 ,names=p_data_set[p_names]
                 ,color=p_names
                 ,hover_data=[p_names]
                 ,labels={p_names: p_names}
                 #,color_discrete_sequence=px.colors.sequential.RdBu
                 ,color_discrete_map = p_color_discrete_map
                 ,hole =p_hole
                 )

    fig_pie.update_layout(title=dict(
                                text="<b>" + f'{p_title}' + "</b>",
                                font=dict(size=fontsize_title_graph_glb),
                                x=0.5,
                                xref="paper"
                                    )
                            ,title_font_color='#346751'
                            ,template='plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
                         )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    return fig_pie

def fn_barchart_color_palette(p_data_set,p_colpatte_list,p_title):
        global fontsize_title_graph_glb
        top_labels = ['Strongly<br>agree', 'Agree', 'Neutral', 'Disagree',
                      'Strongly<br>disagree']
        #top_labels =p_colpatte_list[0]

        colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
                  'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
                  'rgba(190, 192, 213, 1)','rgba(190, 192, 213, 1)',
                  'rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
                  'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
                  'rgba(190, 192, 213, 1)', 'rgba(190, 192, 213, 1)',
                  'rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
                  'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)'
                  ]

        # x_data = [[21, 30, 21, 16],
        #           [24, 31, 19, 15],
        #           [27, 26, 23, 11],
        #           [29, 24, 15, 18],
        #           [29, 24, 15, 18]]
        x_data = [[21, 30, 21, 16],
                  [24, 31, 19, 15],
                  [27, 26, 23, 11],
                  [29, 24, 15, 18],
                  [29, 24, 15, 0]]
        # x_data = [[21, 30, 21, 16],
        #           [24, 31, 19, 15],
        #           [27, 26, 23, 11]]

        # y_data = p_colpatte_list[1]
        # x_data =p_colpatte_list[2]
        y_data = ['The course was effectively<br>organized',
                  'The course developed my<br>abilities and skills ' +
                  'for<br>the subject', 'The course developed ' +
                  'my<br>ability to think critically about<br>the subject',
                  'I would recommend this<br>course to a friend']

        fig = go.Figure()

        for i in range(0, len(x_data[0])):
            for xd, yd in zip(x_data, y_data):
                fig.add_trace(go.Bar(
                    x=[xd[i]], y=[yd],
                    orientation='h',
                    marker=dict(
                        color=colors[i],
                        line=dict(color='rgb(248, 248, 249)', width=1)
                    )
                ))

        fig.update_layout(
            #template = 'plotly_dark'
            xaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
                domain=[0.15, 1]
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=False,
                zeroline=False,
            ),
            barmode='stack',
            paper_bgcolor='rgb(248, 248, 255)',
            plot_bgcolor='rgb(248, 248, 255)',
            #plot_bgcolor='rgb(0, 0, 0)',
            margin=dict(l=120, r=10, t=140, b=80),
            showlegend=False,
        )

        annotations = []

        for yd, xd in zip(y_data, x_data):
            # labeling the y-axis
            annotations.append(dict(xref='paper', yref='y',
                                    x=0.14, y=yd,
                                    xanchor='right',
                                    text=str(yd),
                                    font=dict(family='Arial', size=14,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False, align='right'))
            # labeling the first percentage of each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=xd[0] / 2, y=yd,
                                    text=str(xd[0]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the first Likert scale (on the top)
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=xd[0] / 2, y=1.1,
                                        text=top_labels[0],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space = xd[0]
            for i in range(1, len(xd)):
                # labeling the rest of percentages for each bar (x_axis)
                annotations.append(dict(xref='x', yref='y',
                                        x=space + (xd[i] / 2), y=yd,
                                        text=str(xd[i]) + '%',
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(248, 248, 255)'),
                                        showarrow=False))
                # labeling the Likert scale
                if yd == y_data[-1]:
                    annotations.append(dict(xref='x', yref='paper',
                                            x=space + (xd[i] / 2), y=1.1,
                                            text=top_labels[i],
                                            font=dict(family='Arial', size=14,
                                                      color='rgb(67, 67, 67)'),
                                            showarrow=False))
                space += xd[i]

        # fig.update_layout(annotations=annotations)

        fig.update_layout(title=dict(
                                     text="<b>" + f'{p_title}' + "</b>"
                                     ,font=dict(size=fontsize_title_graph_glb)
                                     ,x=0.5
                                     ,xref="paper"
                                    )
                          ,title_font_color='#346751'
                          ,template='plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
                          ,annotations=annotations
                            )

        return fig

def fn_indicators(p_metrics_dict):
    fig_indicators = go.Figure()
    #print(f"Value ==>{p_value},Ref=>{p_ref_val}")
    #print(p_metrics_dict)
    fig_indicators.add_trace(go.Indicator(
        mode="number+delta"
        #mode = "number+delta+gauge"
        ,value=p_metrics_dict['avg'][1]  #p_value
        ,domain={'x': [0, 0], 'y': [0, 0]}
        ,delta={'reference': p_metrics_dict['avg'][0], 'relative': True, 'position': "top",'valueformat':'.2%','font':{'size':20}}
        ,title={'text':'Promedio','font':{'size': 16}}
        ,number={'font': {'size': 20},'valueformat':'.2f'}
            )
        )

    fig_indicators.add_trace(go.Indicator(
        mode="number+delta"
        , value=p_metrics_dict['mediana'][1]  # p_value
        , domain={'x': [0, 1], 'y': [0, 1]}
        , delta={'reference': p_metrics_dict['mediana'][0], 'relative': True, 'position': "top", 'valueformat': '.2%','font': {'size': 20}}
        , title={'text': 'Mediana', 'font': {'size': 16}}
        , number={'font': {'size': 20}, 'valueformat': '.2f'}
            )
        )

    # fig_indicators.add_trace(go.Indicator(
    #     ,value=p_value
    #     ,delta={'reference': p_ref_val, 'relative': True, 'position': "top", 'valueformat': '.2%', 'font': {'size': 20}}
    #     ,gauge={'axis': {'visible': False}}
    #     ,domain={'row': 0, 'column': 0}
    #     ,number={'font': {'size': 20}, 'valueformat': '.2f'}
    #          )
    #     )

    fig_indicators.update_layout(
                            #width=800,
                            template='plotly_dark'
                            #paper_bgcolor="black"
                            ,height=150
                            ,grid = {'rows': 2, 'columns': 2, 'pattern': "independent"}
                            ,title_text="<b>" + f'KPIs Transaccionales' + "</b>" #'KPIs Transaccionales'
                            ,title_font_color='#346751'
                            #,title_font_size ='20'
                            ,title_y=0.9
                            ,title_x=0.5
                             )

    return fig_indicators

def fn_line_chart(p_data_set,p_title,p_x,p_y,p_color):
    global fontsize_title_graph_glb
    fig_line = px.line(p_data_set, x=p_x, y=p_y,color=p_color, symbol=p_color,markers=False)

    # fig_line.update_layout(
    #     # width=800,
    #     #template='plotly_dark'
    #     , title_text='Volumen Transaccional Mensual por Dia'
    #     , title_font_color='#346751'
    #     , title_y=0.9
    #     , title_x=0.5
    #     )

    fig_line.update_layout(title=dict(
                            text="<b>" + f'{p_title}' + "</b>",
                            font=dict(size=fontsize_title_graph_glb),
                            x=0.5,
                            xref="paper"
                                    )
                            ,title_font_color = '#346751'
                            ,template='plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
                         )

    return fig_line

def fn_graph_sunburst(p_dataset,p_title,p_path,p_values,p_color,p_hover_data,p_avg,p_weights):
    global fontsize_title_graph_glb
    fig_sunburst = px.sunburst(p_dataset
                               ,path=p_path
                               ,values=p_values
                               ,color=p_color
                               ,hover_data=p_hover_data
                               ,color_continuous_scale='RdBu'
                               ,color_continuous_midpoint=np.average(p_dataset[p_avg],weights=p_dataset[p_weights])
                               )

    fig_sunburst.update_layout(title=dict(
                                text="<b>" + f'{p_title}' + "</b>"
                                ,font=dict(size=fontsize_title_graph_glb)
                                ,x=0.5
                                ,xref="paper"
                            )
                                , title_font_color='#346751'
                                , template='plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
                            )

    return fig_sunburst

def fn_stucked_funnel_graph(p_title,p_stucfunn_dat_list):
    global fontsize_title_graph_glb
    fig_tunnel = go.Figure()

    for i in range(len(p_stucfunn_dat_list[1])):
        fig_tunnel.add_trace(go.Funnel(
            name=p_stucfunn_dat_list[1][i],
            y=p_stucfunn_dat_list[0],
            x=p_stucfunn_dat_list[2][i],
            textinfo="value+percent total")
        )

    # fig_tunnel.add_trace(go.Funnel(
    #     name=p_stucfunn_dat_list[1][0],
    #     y=p_stucfunn_dat_list[0],
    #     x=p_stucfunn_dat_list[2][0],
    #     #textinfo="value+percent initial")
    #     textinfo="value+percent total")
    #     )
    #
    # fig_tunnel.add_trace(go.Funnel(
    #     name=p_stucfunn_dat_list[1][1],
    #     orientation="h",
    #     y=p_stucfunn_dat_list[0],
    #     x=p_stucfunn_dat_list[2][1],
    #     textposition="inside",
    #     #textinfo="value+percent previous")
    #     textinfo="value+percent total")
    #     )
    #
    # fig_tunnel.add_trace(go.Funnel(
    #     name=p_stucfunn_dat_list[1][2],
    #     orientation="h",
    #     #y=["Website visit", "Downloads", "Potential customers", "Requested price", "invoice sent", "Finalized"],
    #     y=p_stucfunn_dat_list[0],
    #     x=p_stucfunn_dat_list[2][2],
    #     textposition="inside",
    #     #textposition="outside",
    #     #textinfo="value+percent total")
    #     textinfo="value+percent total")
    #     )

    fig_tunnel.update_layout(title=dict(
                                text="<b>" + f'{p_title}' + "</b>"
                                , font=dict(size=fontsize_title_graph_glb)
                                , x=0.5
                                , xref="paper"
                            )
                                , title_font_color='#346751'
                                , template='plotly_dark'  # others include seaborn, ggplot2, plotly_white, plotly_dark
                            )

    return fig_tunnel

   ## old
    #fig = px.funnel(p_df_data_set, x=p_stucfunn_dat_list[2], y=p_stucfunn_dat_list[0])
    #fig.show()

def fn_get_page_layout(p_key_graph):
    '''Función que define el esqueleto de la pagina a ser presentada en ==> id-page-content'''
    global df_data_bystate, GRAPH_PARAM_DICT
    global df_data_parkterm
    metricts_dict = {}
    if p_key_graph == 'id-graph01-trx-bystate':
        start_date =pd.to_datetime(df_data_bystate['FECHA_TRX'].max(), dayfirst=False).replace(day=1).strftime("%Y-%m-%d")
        end_date   =pd.to_datetime(df_data_bystate['FECHA_TRX'].max(), dayfirst=False).strftime("%Y-%m-%d")
        query_filter= f"FECHA_TRX >='{start_date}' and FECHA_TRX<='{end_date}'"
        df_tbl_data, df_tbl_final, df_graph_pie01, df_barchart_colpltte, metricts_dict,df_data_period = fn_prepare_dataset_page2( p_df_data_filter=df_data_bystate, p_query_filter=query_filter, p_filt='G', p_start_date='2023-07-01', p_bank='All Banks', p_state='All States')

        fig_bar_bystate = fn_barchar_withtext(p_data_set=df_tbl_data
                                              , p_title =GRAPH_PARAM_DICT[p_key_graph][0]
                                              , p_x     =GRAPH_PARAM_DICT[p_key_graph][1]
                                              , p_y     =GRAPH_PARAM_DICT[p_key_graph][2]
                                              , p_color =GRAPH_PARAM_DICT[p_key_graph][3]
                                              , p_text  =GRAPH_PARAM_DICT[p_key_graph][4]
                                              , p_xlabel=GRAPH_PARAM_DICT[p_key_graph][5]
                                              , p_ylabel=GRAPH_PARAM_DICT[p_key_graph][6]
                                              , p_orientacion='v'
                                              , p_color_map  =None
                                              , p_height     =400
                                              , p_pattern_shape=None
                                              , p_pattern_shape_seq_list=None
                                              )

        # Grupo de Graficos de la parte inferior del dashborad
        fig_pie01_graph= fn_get_piechart(p_data_set     =df_graph_pie01
                                         ,p_title       =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][0] # 'VOLUMEN TRANSACCIONAL APROBADAS Vs RECHAZADAS'
                                         ,p_names       =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][1]  #'STATUS_TRX'
                                         ,p_values      =GRAPH_PARAM_DICT['id-graph_pie01-trx-bystate'][2]  #'TOTAL_TRX'
                                         ,p_color_discrete_map={'No Definidas':'lightcyan','Rechazadas':'magenta','Aprobadas':'cyan'}
                                         ,p_hole        =0
                                         )

        fig_indicators =fn_indicators(p_metrics_dict =metricts_dict)
        fig_line       =fn_line_chart(p_data_set   =df_data_period
                                   ,p_title     =GRAPH_PARAM_DICT['id-graph-line_bydate'][0]
                                   ,p_x         =GRAPH_PARAM_DICT['id-graph-line_bydate'][1]    #'FECHA_TRX'
                                   ,p_y         =GRAPH_PARAM_DICT['id-graph-line_bydate'][2]    #'TOTAL_TRX'
                                   ,p_color     =GRAPH_PARAM_DICT['id-graph-line_bydate'][3]    #'STATUS_TRX'
                                   )

        # Layout de la Fila 01
        page_layout_row1 =dbc.Row(
                                html.Div(children=[
                                            # Inicio DropDown Bancos
                                            html.Div(
                                                children=[
                                                    html.Div(children="Bancos", className="menu-title"),
                                                    dcc.Dropdown(id="id-banks-filter",
                                                                 options=[{"label": bank, "value": bank}
                                                                          for bank in banks
                                                                          ],
                                                                 value="All Banks",
                                                                 clearable=False,
                                                                 className="dropdown",
                                                                 ),
                                                ]
                                                , className='c-dropdown-states'
                                            )
                                            # Fin dropdwon Bancos
                                            # Inicio DropDown Estado
                                            ,html.Div(
                                                children=[
                                                    html.Div(children="Estados", className="menu-title"),
                                                    dcc.Dropdown(id="id-states-filter",
                                                                 options=[{"label": state, "value": state}
                                                                          for state in states
                                                                          ],
                                                                 value="All States",
                                                                 clearable=False,
                                                                 className="dropdown",
                                                                 ),
                                                        ]
                                                    ,className='c-dropdown-states'
                                                    )
                                            # Fin dropdwon Estado
                                            # Inicio DatePickerRange
                                            , html.Div(
                                                children=[
                                                    html.Div(children="Date Range", className="menu-title")
                                                    , dcc.DatePickerRange(
                                                        id="id-date-range"
                                                        ,min_date_allowed=pd.to_datetime(df_data_bystate["FECHA_TRX"].min(), dayfirst=False)
                                                        #, min_date_allowed=pd.to_datetime(df_data_bystate["FECHA_TRX"].min(), dayfirst=False).replace(day=1)
                                                        , max_date_allowed=pd.to_datetime(df_data_bystate["FECHA_TRX"].max(), dayfirst=False)
                                                        , start_date      =pd.to_datetime(df_data_bystate["FECHA_TRX"].max(), dayfirst=False).replace(day=1)
                                                        , end_date        =pd.to_datetime(df_data_bystate["FECHA_TRX"].max(),dayfirst=False)
                                                        , display_format  ='DD/MM/YYYY'
                                                        , className       ="dropdown"
                                                    )
                                                ]
                                            )
                                            # Fin DatePickerRange

                                            ]
                                        , className="menu"
                                   )
                                    #,html.H1(children=['KPIs TRANSACCIONALES'], className='c-kpi-h1-title')
                                    )
        # Indicators Page (KPI)
        page_layout_indicators = html.Div(children=[
                                            dbc.Col(
                                                children=[
                                                    dcc.Graph(id='id-graph_ind_bystate', figure=fig_indicators)
                                                        ]
                                                # ,width={'size': 3  , 'offset': 0, 'order': 1}
                                                        )
                                                    ]
                                            , className='c-top-indicators'
                                        )
        # Layout de la Fila 02
        page_layout_row2 =html.Div(children=[html.Div(children=[
                                dbc.Col(
                                       children=[
                                           dash_table.DataTable( id ='id-dash-tbl-states'
                                                                ,data=df_tbl_final.to_dict('records')
                                                                , style_as_list_view=True
                                                                , style_cell={'padding': '5px'
                                                                               , 'textAlign': 'justify'
                                                                               , 'border': '0.1px solid grey'
                                                                              }
                                                                , style_header={
                                                                                   'backgroundColor': 'black'
                                                                                   , 'fontWeight': 'bold'
                                                                                   , 'border': '1px solid #142850'
                                                                                   , 'textAlign': 'center'
                                                                               }
                                                                , style_data={
                                                                                   'backgroundColor': 'rgb(50, 50, 50)'
                                                                                   ,'color': 'white'
                                                                                   ,'border': '1px solid black'

                                                                               }
                                                                , page_size=10
                                                                ,style_data_conditional=[
                                                                                       {
                                                                                           'if': {
                                                                                               'column_id': '%',
                                                                                           },
                                                                                           'backgroundColor': 'dodgerblue'
                                                                                           ,'color': 'white'
                                                                                           ,'textAlign': 'right'
                                                                                       }
                                                                                       ,{
                                                                                           'if': {
                                                                                               'column_id': 'TOTAL_TRX',
                                                                                           },
                                                                                           'textAlign': 'right'
                                                                                       }
                                                                                    ]
                                                                )
                                                ]
                                       #,style={'width': '50%',  'display': 'inline-block'}
                                       ,width={'size': 3, 'offset': 0, 'order': 0,'style': 'font-size: 12px;'}
                                      )
                                ,dbc.Col(
                                       children=[
                                                dcc.Graph(id='id-graph02-trx-bystate',figure=fig_bar_bystate)
                                                ]
                                      #,style = {'width': '50%', 'display': 'inline-block'}
                                        ,className='c-border-graph'
                                        #,style={'padding':'0 1%'}
                                        , width={'size': 9, 'offset': 0, 'order': 1}
                                      )
                                       ]
                                ,className='c-main-box-graph'
                                #,style = {"display": "flex", "border": "double","padding":"1%"}
                              )
                              # Aqui va el segundo parte de graficos
                              ,html.Div(children=[
                                dbc.Col(
                                        children=[
                                            dcc.Graph(id='id-graph_pie01-trx-bystate', figure=fig_pie01_graph)
                                                 ]
                                        , className='c-border-graph'
                                        , width={'size': 6, 'offset': 0, 'order': 1}
                                        )
                                ,dbc.Col(
                                       children=[
                                           dash_table.DataTable(id='id-dash-tbl-tipo-trx'
                                                                , data=df_barchart_colpltte.to_dict('records')
                                                                , style_as_list_view=True
                                                                , style_cell={'padding': '5px'
                                                   , 'textAlign': 'justify'
                                                   , 'border': '0.1px solid grey'
                                                                              }
                                                                , style_header={
                                                   'backgroundColor': 'black'
                                                   , 'fontWeight': 'bold'
                                                   , 'border': '1px solid #142850'
                                                   , 'textAlign': 'center'
                                               }
                                                                , style_data={
                                                   'backgroundColor': 'rgb(50, 50, 50)'
                                                   , 'color': 'white'
                                                   , 'border': '1px solid black'

                                               }
                                                                , page_size=10
                                                                , style_data_conditional=[
                                                   {
                                                       'if': {
                                                           'column_id': '%',
                                                       },
                                                       'backgroundColor': 'dodgerblue'
                                                       , 'color': 'white'
                                                       , 'textAlign': 'right'
                                                   }
                                                   , {
                                                       'if': {
                                                           'column_id': 'TOTAL_TRX',
                                                       },
                                                       'textAlign': 'right'
                                                   }
                                               ]
                                                                )
                                                ]
                                        ,className='c-border-graph'
                                        , width={'size': 6, 'offset': 0, 'order': 1}
                                      )
                                       ]
                                ,className='c-main-box-graph'
                              )
                             # Tercera parte
                             , html.Div(children=[
                                    dbc.Col(
                                        children=[
                                            dcc.Graph(id='id-graph-line_bydate', figure=fig_line)
                                        ]
                                        , className='c-border-graph'
                                        #, width={'size': 6, 'offset': 0, 'order': 1}
                                    )
                                    ]
                                    , className='c-main-box-graph'
                                    )
                            ])

        # page_layout_row3 =html.Div(children=[
        #                                     dbc.Col(
        #                                         children=[
        #                                             dcc.Graph(id='id-graph-line_bydate', figure=fig_line)
        #                                                 ]
        #                                         #, width={'size': 6, 'offset': 0, 'order': 1}
        #                                                 )
        #                                      ]
        #                                     ,className='c-border-graph'
        #                          )

        page_layout_main = dbc.Container(
                                        html.Div([
                                        dbc.Row(children=[page_layout_row1])
                                        ,dbc.Row(children=[page_layout_indicators])
                                        ,dbc.Row(children=[page_layout_row2])
                                        #,dbc.Row(children=[page_layout_row3])
                                                ])
                                        )
    elif p_key_graph == 'id-terminals-park':
        df_grapth_state_tcom,df_data_state_tcom, df_sunburst_datset,df_donut_entadq,df_gbar_colptte,graph_tunel_list,df_pie_tipterm = fn_prepare_dataset_page1(p_df_data_parkterm = df_data_parkterm,p_query_filter=None)

        fig01_bar_term     =fn_barchar_withtext(p_data_set      =df_grapth_state_tcom
                                                ,p_title        =GRAPH_PARAM_DICT['id-graph01-bar-term'][0]
                                                ,p_x            =GRAPH_PARAM_DICT['id-graph01-bar-term'][1]
                                                ,p_y            =GRAPH_PARAM_DICT['id-graph01-bar-term'][2]
                                                ,p_color        =GRAPH_PARAM_DICT['id-graph01-bar-term'][3]
                                                ,p_text         =GRAPH_PARAM_DICT['id-graph01-bar-term'][4]
                                                ,p_xlabel       =GRAPH_PARAM_DICT['id-graph01-bar-term'][5]
                                                ,p_ylabel       =GRAPH_PARAM_DICT['id-graph01-bar-term'][6]
                                                ,p_orientacion  ='v'
                                                ,p_color_map    ={'DialUp': '#9D3C72', 'LAN': '#1F6E8C','Inalambrico': '#4E9F3D','DialUp':'#D21312'}
                                                ,p_height       =500
                                                ,p_pattern_shape =None
                                                ,p_pattern_shape_seq_list=None
                                                )

        fig_sunburst=fn_graph_sunburst(p_dataset    =df_sunburst_datset
                                       ,p_title     =GRAPH_PARAM_DICT['id-terminals-park'][0]
                                       ,p_path      =GRAPH_PARAM_DICT['id-terminals-park'][1]   #['ADQUIRIENCIA','TIPO_TERMINAL','TIPO_COMUNICACION']
                                       ,p_values    =GRAPH_PARAM_DICT['id-terminals-park'][2]   #'CANTIDAD_EQP'
                                       ,p_color     =GRAPH_PARAM_DICT['id-terminals-park'][3]   #'CANTIDAD_EQP'
                                       ,p_hover_data=GRAPH_PARAM_DICT['id-terminals-park'][4]   #['TIPO_TERMINAL','TIPO_COMUNICACION']
                                       ,p_avg       =GRAPH_PARAM_DICT['id-terminals-park'][5]   #'CANTIDAD_EQP'
                                       ,p_weights   =GRAPH_PARAM_DICT['id-terminals-park'][6]   #'CANTIDAD_EQP'
                                       )
        #print(df_donut_entadq.head(5))
        fig_donut_entadq = fn_get_piechart(p_data_set   =df_donut_entadq
                                          ,p_title     = GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][0]
                                          ,p_names     = GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][1]
                                          ,p_values    = GRAPH_PARAM_DICT['id-graph01-donuts-entadq'][2]
                                          ,p_color_discrete_map={'BANCO PROVINCIAL': 'lightcyan', 'BANCO MERCANTIL': '#205295','COMUN': '#CD1818'}
                                           ,p_hole      =0.4
                                          )

        # fig_bar_colptte = fn_barchart_color_palette(p_data_set          =df_gbar_colptte
        #                                             ,p_colpatte_list    =graph_colptte_list
        #                                             ,p_title            ='ADQUIRIENCIA y TIPO DE COMUNICACIÓN')

        fig_tunnel=fn_stucked_funnel_graph(p_title=GRAPH_PARAM_DICT['id-graph01-bar-tunnel'][0]
                                           ,p_stucfunn_dat_list=graph_tunel_list)

        # fig01_pie_tipterm   = fn_get_piechart(p_data_set   =df_pie_tipterm
        #                                    , p_title    =GRAPH_PARAM_DICT['id-graph01-tipo-term'][0]
        #                                    , p_names    =GRAPH_PARAM_DICT['id-graph01-tipo-term'][1]
        #                                    , p_values   =GRAPH_PARAM_DICT['id-graph01-tipo-term'][2]
        #                                    , p_color_discrete_map={'BANCO PROVINCIAL': 'lightcyan','BANCO MERCANTIL': '#205295', 'COMUN': '#CD1818'}
        #                                    , p_hole     =0
        #                                    )
        fig01_pie_tipterm =fn_barchar_withtext(p_data_set   =df_pie_tipterm
                                                , p_title   =GRAPH_PARAM_DICT['id-graph01-tipo-term'][0]
                                                , p_x       =GRAPH_PARAM_DICT['id-graph01-tipo-term'][1]
                                                , p_y       =GRAPH_PARAM_DICT['id-graph01-tipo-term'][2]
                                                , p_color   =GRAPH_PARAM_DICT['id-graph01-tipo-term'][3]
                                                , p_text    =GRAPH_PARAM_DICT['id-graph01-tipo-term'][4]
                                                , p_xlabel  =GRAPH_PARAM_DICT['id-graph01-tipo-term'][5]
                                                , p_ylabel  =GRAPH_PARAM_DICT['id-graph01-tipo-term'][6]
                                                , p_orientacion='v'
                                                , p_color_map={'CAJA REGIST.': '#9D3C72', 'POS': '#1F6E8C'}
                                                , p_height=400
                                                , p_pattern_shape=GRAPH_PARAM_DICT['id-graph01-tipo-term'][4]
                                                ,p_pattern_shape_seq_list=[".", "x"]
                                                )
        ## Deinición del Layout de paginas

        page_layout_row1 = dbc.Row(
                                    html.Div(children=[
                                        # Inicio DropDown Bancos
                                        html.Div(
                                            children=[
                                                html.Div(children="Bancos", className="menu-title"),
                                                dcc.Dropdown(id="id-banks-filter",
                                                             options=[{"label": bank, "value": bank}
                                                                      for bank in banks_pag1
                                                                      ],
                                                             value="All Banks",
                                                             clearable=False,
                                                             className="dropdown",
                                                             ),
                                            ]
                                            , className='c-dropdown-states'
                                        )
                                        # Fin dropdwon Bancos
                                        # # Inicio DropDown Tipo de Terminales
                                        # , html.Div(
                                        #     children=[
                                        #         html.Div(children="Tipo de terminal", className="menu-title"),
                                        #         dcc.Dropdown(id="id-tipterm-filter",
                                        #                      options=[{"label": tip_term, "value": tip_term}
                                        #                               for tip_term in tip_term_pag1
                                        #                               ],
                                        #                      value="All Terminals",
                                        #                      clearable=False,
                                        #                      className="dropdown",
                                        #                      ),
                                        #     ]
                                        #     , className='c-dropdown-states'
                                        # )
                                        # # Fin dropdwon tipo de terminales
                                    ]
                                        , className="menu"
                                    )
                                    # ,html.H1(children=['KPIs TRANSACCIONALES'], className='c-kpi-h1-title')
                                )

        page_layout_row2=html.Div(children=[
                                            dbc.Col(children=[
                                                dash_table.DataTable(id='id-dash-tbl01-term'
                                                                     , data=df_data_state_tcom.to_dict('records')
                                                                     , style_as_list_view=True
                                                                     , style_cell={'padding': '5px'
                                                        , 'textAlign': 'justify'
                                                        , 'border': '0.1px solid grey'
                                                                                   }
                                                                     , style_header={
                                                        'backgroundColor': 'black'
                                                        , 'fontWeight': 'bold'
                                                        , 'border': '1px solid #142850'
                                                        , 'textAlign': 'center'
                                                    }
                                                                     , style_data={
                                                        'backgroundColor': 'rgb(50, 50, 50)'
                                                        , 'color': 'white'
                                                        , 'border': '1px solid black'

                                                    }
                                                                     , page_size=10
                                                                     , style_data_conditional=[
                                                        {
                                                            'if': {
                                                                'column_id': '%',
                                                            },
                                                            'backgroundColor': 'dodgerblue'
                                                            , 'color': 'white'
                                                            , 'textAlign': 'right'
                                                        }
                                                        , {
                                                            'if': {
                                                                'column_id': 'CANTIDAD_EQP',
                                                            },
                                                            'textAlign': 'right'
                                                        }
                                                    ]
                                                                     )
                                                ,dbc.Col(children=[
                                                            dcc.Graph(id='id-graph02-sunburst', figure=fig_sunburst)
                                                                    ]
                                                        , className='c-border-graph'
                                                         )
                                                             ]
                                                   ,className='c-border-graph'
                                                   ,style = {'display': 'flex'}
                                                    )
                                                 ,dbc.Col(children=[
                                                                    dcc.Graph(id='id-graph01-bar-term', figure=fig01_bar_term)
                                                                    ,dcc.Graph(id='id-graph01-bar-tunnel', figure=fig_tunnel)

                                                                    # , width={'size': 8, 'offset': 0, 'order': 0}

                                                                # ,dbc.Col(children=[
                                                                #             dcc.Graph(id='id-graph01-tipo-term', figure=fig01_pie_tipterm)
                                                                #                 ]
                                                                #             ,width={'size': 5, 'offset': 0, 'order': 1}
                                                                #          )
                                                                    ]
                                                            , className='c-border-graph'
                                                            #, style={'display': 'flex'}
                                                            #,width={'size': 12, 'offset': 0, 'order': 1}
                                                         )
                                                ]
                                        , className='c-main-box-graph_pag1'
                                 )

        page_layout_row3 = html.Div(children=[
                                    dbc.Col(children=[
                                        dbc.Col(children=[
                                                    dcc.Graph(id='id-graph01-donuts-entadq', figure=fig_donut_entadq)
                                                            ]
                                                    , className='c-border-graph'
                                                    #,width={'size': 5, 'offset': 0, 'order': 1}
                                                  )
                                        ,dbc.Col(children=[
                                                    dcc.Graph(id='id-graph01-tipo-term', figure=fig01_pie_tipterm)
                                                       ]
                                             , className='c-border-graph'
                                             #,width={'size': 6, 'offset': 0, 'order': 2}
                                             )
                                                    ]
                                                , className='c-border-graph'
                                                , style={'display': 'flex'}
                                            )
                                    # ,dbc.Col(children=[
                                    #             dcc.Graph(id='id-graph01-bar-colptte', figure=fig_bar_colptte)
                                    #                   ]
                                    #         , className='c-border-graph'
                                    #         )
                                            ]
                                    , className='c-main-box-graph_pag1'
                                    )

        page_layout_main = dbc.Container(
                                    html.Div(children=[
                                                 dbc.Row(children=[page_layout_row1])
                                                ,dbc.Row(children=[page_layout_row2])
                                                ,dbc.Row(children=[page_layout_row3])
                                                      ]
                                            )
                                        )

    else:
        # Definir el Layout de la página
        page_layout_main =html.Div(children=[html.P('PAGINA PRINCIPAL')])

    return page_layout_main

# Run the app
if __name__ == '__main__':
    app.run(debug=True)