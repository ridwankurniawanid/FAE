import os
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import base64
import comtrade
import numpy as np
import plotly.graph_objects as go
from tabulate import tabulate  # Import tabulate for table formatting

# Ensure uploads directory exists
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Custom style definitions
CUSTOM_STYLES = '''
    .large-checkbox input[type="checkbox"] {
        width: 20px;
        height: 20px;
        margin-right: 10px;
    }
    .large-checkbox label {
        display: inline-flex;
        align-items: center;
        font-size: 16px;
    }
    .channel-table {
        max-width: 600px;
        margin: 0 auto;
    }
    .select-channel-btn {
        margin-top: 15px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
'''

# Initialize the Dash app with Bootstrap and custom CSS
app = dash.Dash(__name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        {'href': '/assets/custom.css', 'rel': 'stylesheet'}
    ],
    suppress_callback_exceptions=True
)

server=app.server
# Ensure the custom CSS file exists in the assets folder
os.makedirs('assets', exist_ok=True)
with open('assets/custom.css', 'w') as f:
    f.write(CUSTOM_STYLES)

# Global variables to store uploaded files and channels
CFG_FILE = None
cfgcsv = None
DAT_FILE = None
ANALOG_CHANNELS = []
LEFT_CHANNELS = []
RIGHT_CHANNELS = []
COMTRADE_RECORD = None
signaldf = None
significant = None  # Global variable untuk menyimpan hasil perhitungan
sampling = 1000  # Set your sampling frequency here

# Function to process selected channels
def process_selected_channels(selected_channels):
    global COMTRADE_RECORD, cfgcsv, signaldf
    
    if COMTRADE_RECORD is None:
        print("No Comtrade record loaded.")
        return None
    
    processed_channels = {}
    
    for channel_id in selected_channels:
        try:
            channel_index = ANALOG_CHANNELS.index(channel_id)
            channel_data = COMTRADE_RECORD.analog[channel_index]
            processed_channels[f'var{channel_index}'] = channel_data
            
            if cfgcsv.iloc[channel_index, 4] == "kA":
                processed_channels[f'var{channel_index}'] = np.multiply(channel_data, 1000)
            if cfgcsv.iloc[channel_index, 4] == "V":
                processed_channels[f'var{channel_index}'] = np.divide(channel_data, 1000)
        
        except ValueError:
            print(f"Channel {channel_id} not found in analog channels.")
        except Exception as e:
            print(f"Error processing channel {channel_id}: {str(e)}")
    
    if processed_channels:
        signaldf = pd.DataFrame(processed_channels)
        signaldf.columns = ['VR', 'VS', 'VT','IR', 'IS', 'IT','IN']    
        return signaldf
    
    return None

# Function to calculate significant change
def signichange(signaldfull):
    global significant
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(signaldfull)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    diff_variance = np.diff(pca_df['Principal Component 1'])
    threshold = 0.1  
    significant_change_index = np.where(diff_variance < -threshold)[0]
    significant = significant_change_index[0] if len(significant_change_index) > 0 else None
    return significant

# Function to calculate THD (Total Harmonic Distortion)
def THDCALC(signal):
    N = len(signal)
    fs = 25600
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(fft_result)[:N//2]
    fundamental_freq_index = np.where(frequencies >=50)[0][0]
    A1 = magnitude[fundamental_freq_index]
    harmonic_indices = [2 * fundamental_freq_index, 3 * fundamental_freq_index, 4 * fundamental_freq_index,
                        5 * fundamental_freq_index, 6 * fundamental_freq_index, 7 * fundamental_freq_index,
                        8 * fundamental_freq_index, 9 * fundamental_freq_index]
    A_harmonics = [magnitude[i] for i in harmonic_indices if i < len(magnitude)]
    
    THD = np.sqrt(sum(A**2 for A in A_harmonics)) / A1
    return THD

# Function to display table
def display_table(significant, signaldf):
    VRBefore = signaldf['VR'][0:significant-1]
    VSBefore = signaldf['VS'][0:significant-1]
    VTBefore = signaldf['VT'][0:significant-1]
    IRBefore = signaldf['IR'][0:significant-1]
    ISBefore = signaldf['IS'][0:significant-1]
    ITBefore = signaldf['IT'][0:significant-1]
    INBefore = signaldf['IN'][0:significant-1]
    
    VRAfter = signaldf['VR'][significant:significant+1024]
    VSAfter = signaldf['VS'][significant:significant+1024]
    VTAfter = signaldf['VT'][significant:significant+1024]
    IRAfter = signaldf['IR'][significant:significant+1024]
    ISAfter = signaldf['IS'][significant:significant+1024]
    ITAfter = signaldf['IT'][significant:significant+1024]
    INAfter = signaldf['IN'][significant:significant+1024]
    
    VRBeforerms = np.sqrt(np.mean(VRBefore**2))
    VSBeforerms = np.sqrt(np.mean(VSBefore**2))
    VTBeforerms = np.sqrt(np.mean(VTBefore**2))
    
    VRAfterrms = np.sqrt(np.mean(VRAfter**2))
    VSAfterrms = np.sqrt(np.mean(VSAfter**2))
    VTAfterrms = np.sqrt(np.mean(VTAfter**2))
    
    IRBeforerms = np.sqrt(np.mean(IRBefore**2))
    ISBeforerms = np.sqrt(np.mean(ISBefore**2))
    ITBeforerms = np.sqrt(np.mean(ITBefore**2))
    INBeforerms = np.sqrt(np.mean(INBefore**2))
    
    IRAfterrms = np.sqrt(np.mean(IRAfter**2))
    ISAfterrms = np.sqrt(np.mean(ISAfter**2))
    ITAfterrms = np.sqrt(np.mean(ITAfter**2))
    INAfterrms = np.sqrt(np.mean(INAfter**2))
    
    THDVRB = THDCALC(VRBefore[0:1024]) * 100
    THDVSB = THDCALC(VSBefore[0:1024]) * 100
    THDVTB = THDCALC(VTBefore[0:1024]) * 100
    THDVRA = THDCALC(signaldf['VR'][significant-512:significant+2048]) * 100
    THDVSA = THDCALC(signaldf['VS'][significant-512:significant+2048]) * 100
    THDVTA = THDCALC(signaldf['VT'][significant-512:significant+2048]) * 100
    
    THDIRB = THDCALC(IRBefore[0:1024]) * 100
    THDISB = THDCALC(ISBefore[0:1024]) * 100
    THDITB = THDCALC(ITBefore[0:1024]) * 100
    THDINB = THDCALC(INBefore[0:1024]) * 100
    THDIRA = THDCALC(IRAfter[0:1024]) * 100
    THDISA = THDCALC(ISAfter[0:1024]) * 100
    THDITA = THDCALC(ITAfter[0:1024]) * 100
    THDINA = THDCALC(INAfter[0:1024]) * 100
    
    headers = ['Calculation', 'VR (kV)', 'VS (kV)', 'VT (kV)', 'IR (A)', 'IS (A)', 'IT (A)', 'IN (A)']
    datatabel = [
        ['Pre Fault (rms)', VRBeforerms, VSBeforerms, VTBeforerms, IRBeforerms, ISBeforerms, ITBeforerms, INBeforerms],
        ['Fault 2 Cycle (rms)', VRAfterrms, VSAfterrms, VTAfterrms, IRAfterrms, ISAfterrms, ITAfterrms, INAfterrms],
        ['Changes (%)',
         (VRAfterrms - VRBeforerms) / VRBeforerms * 100,
         (VSAfterrms - VSBeforerms) / VSBeforerms * 100,
         (VTAfterrms - VTBeforerms) / VTBeforerms * 100,
         (IRAfterrms - IRBeforerms) / IRBeforerms * 100,
         (ISAfterrms - ISBeforerms) / ISBeforerms * 100,
         (ITAfterrms - ITBeforerms) / ITBeforerms * 100,
         (INAfterrms - INBeforerms) / INBeforerms * 100
        ],
        ['DC Offset (DC)', np.mean(VRAfter), np.mean(VSAfter), np.mean(VTAfter), np.mean(IRAfter), np.mean(ISAfter), np.mean(ITAfter), np.mean(INAfter)],
        ['THD Pre Fault(%)', THDVRB, THDVSB, THDVTB, THDIRB, THDISB, THDITB, THDINB],
        ['THD Fault(%)', THDVRA, THDVSA, THDVTA, THDIRA, THDISA, THDITA, THDINA]
    ]
    
    table = tabulate(datatabel, headers, tablefmt='grid')
    return table

# App layout
app.layout = dbc.Container([
    dbc.Tabs([
        # Tab 1: File Upload and Channel Selection
        dbc.Tab(label="File Upload", children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                multiple=True,
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                accept='.cfg,.dat'
            ),
            html.Div(id='output-data-upload'),
            html.Div(id='channel-display'),
            html.Div(id='channel-checkbox-container', style={'marginTop': '20px'}),
            html.Div(
                dbc.Button(
                    "Select Channels", 
                    id='select-channels-btn', 
                    color='primary', 
                    className='select-channel-btn',
                    disabled=True
                ), 
                style={'textAlign': 'center'}
            ),
            html.Div(id='selected-channels-output', style={'marginTop': '20px'}),
            html.Div(id='processed-data-display', style={'marginTop': '20px'})
        ]),
        
        # Tab 2: Data Visualization
        dbc.Tab(label="Data Visualization", children=[
            html.H3("Visualize Comtrade Data"),
            dbc.Button("Plot Data", id="plot-data-btn", color="primary", className="mt-3"),
            html.Div(id="plot-output")
        ]),
        
        # Tab 3: Export
        dbc.Tab(label="Export", children=[
            html.H3("Export Processed Data"),
            dbc.Button(
                "Calculate Significant Change", 
                id="calculate-btn", 
                color="primary", 
                className="mt-3"
            ),
            html.Div(id="calculate-output", className="mt-3")
        ])
    ])
], className='p-4')

# Callback untuk upload file
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('channel-display', 'children'),
     Output('channel-checkbox-container', 'children'),
     Output('select-channels-btn', 'disabled')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_output(list_of_contents, list_of_names):
    global CFG_FILE, DAT_FILE, ANALOG_CHANNELS, LEFT_CHANNELS, RIGHT_CHANNELS, COMTRADE_RECORD, cfgcsv
    
    if list_of_contents is None:
        return [], [], [], True
    
    children = []
    channel_info = []
    channel_checkboxes = []
    btn_disabled = True
    
    for contents, filename in zip(list_of_contents, list_of_names):
        if filename.endswith('.cfg') or filename.endswith('.dat'):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            save_path = os.path.join(UPLOAD_DIRECTORY, filename)
            
            with open(save_path, 'wb') as f:
                f.write(decoded)
            
            if filename.endswith('.cfg'):
                CFG_FILE = save_path
            elif filename.endswith('.dat'):
                DAT_FILE = save_path
            
            children.append(html.Div(f"Uploaded: {filename}"))
    
    if CFG_FILE and DAT_FILE:
        try:
            COMTRADE_RECORD = comtrade.load(CFG_FILE, DAT_FILE)
            cfgcsv = pd.read_csv(CFG_FILE, skiprows=2, header=None)
            ANALOG_CHANNELS = COMTRADE_RECORD.analog_channel_ids
            LEFT_CHANNELS = ANALOG_CHANNELS[:9]
            RIGHT_CHANNELS = ANALOG_CHANNELS[9:18]
            
            table_data = []
            max_length = max(len(LEFT_CHANNELS), len(RIGHT_CHANNELS))
            
            for i in range(max_length):
                row = []
                if i < len(LEFT_CHANNELS):
                    left_ch = LEFT_CHANNELS[i]
                    left_checkbox = dcc.Checklist(
                        id={'type': 'channel-checkbox', 'index': f'left-{left_ch}'},
                        options=[{'label': f' Left Ch {left_ch}', 'value': left_ch}],
                        value=[],
                        className='large-checkbox',
                        inline=True
                    )
                    row.append(html.Td(left_checkbox, style={'width': '50%'}))
                else:
                    row.append(html.Td(style={'width': '50%'}))
                
                if i < len(RIGHT_CHANNELS):
                    right_ch = RIGHT_CHANNELS[i]
                    right_checkbox = dcc.Checklist(
                        id={'type': 'channel-checkbox', 'index': f'right-{right_ch}'},
                        options=[{'label': f' Right Ch {right_ch}', 'value': right_ch}],
                        value=[],
                        className='large-checkbox',
                        inline=True
                    )
                    row.append(html.Td(right_checkbox, style={'width': '50%'}))
                else:
                    row.append(html.Td(style={'width': '50%'}))
                
                table_data.append(html.Tr(row))
            
            channel_table = html.Table(
                [html.Thead(html.Tr([
                    html.Th("Left Channels", style={'width': '50%', 'text-align': 'center'}), 
                    html.Th("Right Channels", style={'width': '50%', 'text-align': 'center'})
                ]))] +
                [html.Tbody(table_data)],
                className='table table-bordered table-striped channel-table'
            )
            channel_checkboxes.append(channel_table)
            btn_disabled = False
            
        except Exception as e:
            channel_info.append(html.Div(f"Error processing Comtrade files: {str(e)}"))
    
    return children, channel_info, channel_checkboxes, btn_disabled

# Callback untuk select channels
@app.callback(
    [Output('selected-channels-output', 'children'),
     Output('processed-data-display', 'children')],
    Input('select-channels-btn', 'n_clicks'),
    [State({'type': 'channel-checkbox', 'index': dash.ALL}, 'value')],
    prevent_initial_call=True
)
def select_channels(n_clicks, checkbox_values):
    global signaldf

    if n_clicks is None:
        return [html.Div("Please click 'Select Channels' to update")], None
    
    selected_channels = []
    for values in checkbox_values:
        if values:
            selected_channels.extend(values)
            
    left_selected = [ch for ch in selected_channels if ch in LEFT_CHANNELS]
    right_selected = [ch for ch in selected_channels if ch in RIGHT_CHANNELS]
    
    selection_details = [
        html.Div([
            html.H5("Selection Details", className="mt-3"),
            html.Div([
                html.Strong("Left Channels: "),
                html.Span(", ".join(map(str, left_selected)) if left_selected else "None")
            ]),
            html.Div([
                html.Strong("Right Channels: "),
                html.Span(", ".join(map(str, right_selected)) if right_selected else "None")
            ]),
            html.Div([
                html.Strong("Total Selected: "),
                html.Span(str(len(selected_channels)))
            ])
        ], className="mt-3 p-3 border rounded bg-light")
    ]
    
    # Process selected channels and display data
    processed_df = process_selected_channels(selected_channels)
    signaldf = processed_df
    
    # Display processed data if available
    if processed_df is not None:
        data_display = [
            html.Div([
                html.Strong("Data Shape: "),
                html.Span(f"{processed_df.shape[0]} rows, {processed_df.shape[1]} columns")
            ])
        ]
    else:
        data_display = [
            html.Div("No data could be processed.", className="alert alert-warning")
        ]
    
    return selection_details, data_display

# Callback untuk plot data
@app.callback(
    Output("plot-output", "children"),
    Input("plot-data-btn", "n_clicks"),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    global signaldf
    
    if n_clicks is None or signaldf is None:
        return None

    # Define 7 distinct colors in hex format
    colors = [
        '#d62728',  # muted blue
        '#ff7f0e',  # safety orange
        '#1f77b4',  # cooked asparagus green
        '#d62728',  # brick red
        '#ff7f0e',  # muted purple
        '#1f77b4',  # chestnut brown
        'black'   # raspberry yogurt pink
    ]

    graphs = []
    
    for idx, col in enumerate(signaldf.columns):
        # Cycle through colors if there are more than 7 columns
        color = colors[idx % len(colors)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=signaldf.index,
            y=signaldf[col],
            mode='lines',
            name=col,
            line=dict(width=2, color=color)
        ))

        # Update layout
        fig.update_layout(
            template='ggplot2',
            title=f"Signal Plot - {col}",
            xaxis_title="Index",
            yaxis_title=f"{col}",
            margin={"t": 40, "b": 40, "l": 40, "r": 40},
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            )
        )

        graphs.append(
            dcc.Graph(
                figure=fig,
                id=f'graph-{col}',
                responsive=True,
                style={
                    'width': '100%',
                    'height': '45vh',
                    'min-height': '400px'
                }
            )
        )

    return html.Div(
        graphs,
        style={
            'margin': '20px auto',
            'display': 'flex',
            'flexDirection': 'column',
            'gap': '30px',
            'maxWidth': '1200px',
            'width': '100%'
        }
    )

# Callback untuk tombol Calculate di Tab3
@app.callback(
    Output("calculate-output", "children"),
    Input("calculate-btn", "n_clicks"),
    prevent_initial_call=True
)
def calculate_significant(n_clicks):
    global signaldf, significant
    if n_clicks is None:
        return dash.no_update
    
    if signaldf is None:
        return html.Div("No data available. Please process channels first.", className="alert alert-danger")
    
    try:
        result = signichange(signaldf)
        table_output = display_table(result, signaldf) if result is not None else "No significant change detected."
        return html.Div([
            html.H5("Calculation Result"),
            html.P(f"Significant Change Index: {result}", className="lead"),
            html.Pre(table_output, style={"white-space": "pre-wrap"})  # Display the table
        ],style={"font-size": "20px"})#, className="alert alert-success")
    except Exception as e:
        return html.Div(f"Error during calculation: {str(e)}", className="alert alert-danger")

# Main app runner
if __name__ == '__main__':
    app.run(debug=False)
