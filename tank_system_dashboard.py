import numpy as np
from scipy.integrate import solve_ivp
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Tank system parameters
A1, A2 = 154, 154
a1, a2, a12 = 0.7498, 0.8040, 0.2445
k1, k2 = 33.336, 25.002
beta1, beta2, beta_x = 0.5, 0.5, 0.5
g = 981

# Initial and default setpoints
h1_0, h2_0 = 24.6, 14.4
default_setpoint_h1, default_setpoint_h2 = 30.0, 20.0

# Simulation parameters
duration = 200
num_points = 500
dt = duration / num_points

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, output_limits=(0,5)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral = 0
        self.prev_error = 0

    def compute(self, measured, dt):
        error = self.setpoint - measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output_clamped = max(self.output_limits[0], min(output, self.output_limits[1]))
        if output != output_clamped:
            self.integral -= error * dt
        self.prev_error = error
        return output_clamped

def tank_system(t, h, u1, u2):
    h1, h2 = h
    disturbance1 = np.random.normal(0, 0.1) if int(t) % 50 == 0 else 0
    disturbance2 = np.random.normal(0, 0.1) if int(t) % 70 == 0 else 0
    dh1_dt = (
        k1 * u1
        - beta1 * a1 * np.sqrt(2 * g * h1)
        - beta_x * a12 * np.sqrt(2 * g * max(h1 - h2, 0))
    ) / A1 + disturbance1
    dh2_dt = (
        k2 * u2
        + beta_x * a12 * np.sqrt(2 * g * max(h1 - h2, 0))
        - beta2 * a2 * np.sqrt(2 * g * h2)
    ) / A2 + disturbance2
    return [dh1_dt, dh2_dt]

def simulate_system(pid1, pid2, noise_std=0):
    t_eval = np.linspace(0, duration, num_points)
    h1, h2 = h1_0, h2_0
    h1_values, h2_values, u1_values, u2_values = [], [], [], []

    for t in t_eval:
        measured_h1 = h1 + np.random.normal(0, noise_std)
        measured_h2 = h2 + np.random.normal(0, noise_std)

        u1 = pid1.compute(measured_h1, dt)
        u2 = pid2.compute(measured_h2, dt)

        h1_values.append(h1)
        h2_values.append(h2)
        u1_values.append(u1)
        u2_values.append(u2)

        h_next = solve_ivp(
            tank_system, [t, t + dt], [h1, h2], args=(u1, u2), method="RK45"
        ).y
        h1, h2 = h_next[:, -1]

    return t_eval, h1_values, h2_values, u1_values, u2_values

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Two-Tank System Simulation with Tank Fill Comparison",
            style={'color': 'white', 'backgroundColor': '#4CAF50', 'padding': '10px'}),

    html.Div([
        html.Div([
            html.H3("Control Panel"),
            html.Label("Setpoint Tank 1 (cm):"),
            dcc.Input(id='setpoint1', type='number', value=default_setpoint_h1),
            html.Label("Setpoint Tank 2 (cm):"),
            dcc.Input(id='setpoint2', type='number', value=default_setpoint_h2),
            html.Label("PID Gains Tank 1 (Kp, Ki, Kd):"),
            dcc.Input(id='kp1', type='number', value=2.0),
            dcc.Input(id='ki1', type='number', value=0.5),
            dcc.Input(id='kd1', type='number', value=0.1),
            html.Label("PID Gains Tank 2 (Kp, Ki, Kd):"),
            dcc.Input(id='kp2', type='number', value=2.0),
            dcc.Input(id='ki2', type='number', value=0.5),
            dcc.Input(id='kd2', type='number', value=0.1),
            html.Label("Sensor Noise Std Dev:"),
            dcc.Input(id='noise_std', type='number', value=0.2),
            html.Label("Theme:"),
            dcc.Dropdown(
                id='theme-selector',
                options=[
                    {'label': 'Light', 'value': 'light'},
                    {'label': 'Dark', 'value': 'dark'}
                ],
                value='light'
            ),
            html.Button("Run Simulation", id="run-button")
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

        html.Div([
            dcc.Graph(id='tank-comparison'),
            dcc.Graph(id='tank1-level'),
            dcc.Graph(id='tank2-level'),
            dcc.Graph(id='pump1-voltage'),
            dcc.Graph(id='pump2-voltage'),
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

@app.callback(
    [
        Output('tank-comparison', 'figure'),
        Output('tank1-level', 'figure'),
        Output('tank2-level', 'figure'),
        Output('pump1-voltage', 'figure'),
        Output('pump2-voltage', 'figure')
    ],
    Input('run-button', 'n_clicks'),
    State('setpoint1', 'value'),
    State('setpoint2', 'value'),
    State('kp1', 'value'),
    State('ki1', 'value'),
    State('kd1', 'value'),
    State('kp2', 'value'),
    State('ki2', 'value'),
    State('kd2', 'value'),
    State('noise_std', 'value'),
    State('theme-selector', 'value')
)
def update_graphs(n_clicks, setpoint1, setpoint2, kp1, ki1, kd1, kp2, ki2, kd2, noise_std, theme):
    pid1 = PIDController(kp1, ki1, kd1, setpoint1)
    pid2 = PIDController(kp2, ki2, kd2, setpoint2)
    t_eval, h1, h2, u1, u2 = simulate_system(pid1, pid2, noise_std)
    template = 'plotly_dark' if theme == 'dark' else 'plotly'

    # Comparison bar chart (final levels)
    fig_tank = go.Figure(go.Bar(
        x=["Tank 1", "Tank 2"],
        y=[h1[-1], h2[-1]],
        marker=dict(color=['blue', 'green']),
        text=[f"{h1[-1]:.1f} cm", f"{h2[-1]:.1f} cm"],
        textposition="outside"
    ))
    fig_tank.update_layout(template=template, title="Tank Liquid Levels",
                           yaxis=dict(title="Height (cm)", range=[0, max(max(h1), max(h2))*1.2]))

    fig1 = go.Figure(go.Scatter(x=t_eval, y=h1, name="Tank 1 Level"))
    fig1.update_layout(template=template, title="Tank 1 Level Over Time",
                       xaxis_title="Time (s)", yaxis_title="Height (cm)")

    fig2 = go.Figure(go.Scatter(x=t_eval, y=h2, name="Tank 2 Level"))
    fig2.update_layout(template=template, title="Tank 2 Level Over Time",
                       xaxis_title="Time (s)", yaxis_title="Height (cm)")

    fig3 = go.Figure(go.Scatter(x=t_eval, y=u1, name="Pump 1 Voltage"))
    fig3.update_layout(template=template, title="Pump 1 Voltage",
                       xaxis_title="Time (s)", yaxis_title="Voltage (V)")

    fig4 = go.Figure(go.Scatter(x=t_eval, y=u2, name="Pump 2 Voltage"))
    fig4.update_layout(template=template, title="Pump 2 Voltage",
                       xaxis_title="Time (s)", yaxis_title="Voltage (V)")

    return fig_tank, fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True)
