import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .state import State


def plot_state(t, y):
    """Plot all true states from the airframe."""
    fig = make_subplots(
        6,
        2,
        shared_xaxes=True,
        x_title="Time (s)",
        subplot_titles=(
            "Position X",
            "Velocity X",
            "Position Y",
            "Velocity Y",
            "Position Z",
            "Velocity Z",
            "Phi",
            "Phi dot",
            "Theta",
            "Theta dot",
            "Psi",
            "Psi dot",
        ),
    )

    fig.add_scatter(
        x=t, y=y[0], row=1, col=1, name="Position X",
    )
    fig.add_scatter(
        x=t, y=y[1], row=2, col=1, name="Position Y",
    )
    fig.add_scatter(
        x=t, y=y[2], row=3, col=1, name="Position Z",
    )
    fig.add_scatter(
        x=t, y=y[3], row=1, col=2, name="Velocity X",
    )
    fig.add_scatter(
        x=t, y=y[4], row=2, col=2, name="Velocity Y",
    )
    fig.add_scatter(
        x=t, y=y[5], row=3, col=2, name="Velocity Z",
    )
    fig.add_scatter(
        x=t, y=y[6], row=4, col=1, name="Phi",
    )
    fig.add_scatter(
        x=t, y=y[7], row=5, col=1, name="Theta",
    )
    fig.add_scatter(
        x=t, y=y[8], row=6, col=1, name="Psi",
    )
    fig.add_scatter(
        x=t, y=y[9], row=4, col=2, name="Phi dot",
    )
    fig.add_scatter(
        x=t, y=y[10], row=5, col=2, name="Theta dot",
    )
    fig.add_scatter(
        x=t, y=y[11], row=6, col=2, name="Psi dot",
    )

    return fig


def body_to_ned(point, state: State):
    """Convert point from body frame to north-east-down."""
    phi, theta, psi = state.angle
    R_roll = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)],]
    )
    R_pitch = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    R_yaw = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1],]
    )
    R = R_roll @ R_pitch @ R_yaw
    ned = R @ point

    return ned


def ned_to_enu(point):
    """Convert point from north-east-down to east-north-up."""
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1],])
    enu = R @ point
    return enu


def render_aircraft_frame(state, body_vertices):
    """Render an aircraft frame state from its body frame vertices to a plot trace."""
    points_enu = np.row_stack(
        (
            ned_to_enu(body_to_ned(point, state) + state.position)
            for point in body_vertices
        )
    )

    trace = go.Mesh3d(
        x=points_enu[:, 0], y=points_enu[:, 1], z=points_enu[:, 2], color="blue",
    )

    return trace


def animate_airframe(time, state_array, airframe_vertices):
    """Animate airframe postion over time."""

    state_series = [
        State.from_vector(np.hstack((t, state_y)))
        for t, state_y in zip(time, state_array.T)
    ]

    frames = [render_aircraft_frame(state, airframe_vertices) for state in state_series]

    body_animation = go.Figure(
        data=[frames[0]],
        layout=go.Layout(
            title="Aircraft Animation",
            scene=dict(
                xaxis=dict(range=[-50, 50], autorange=False),
                yaxis=dict(range=[0, 100], autorange=False),
                zaxis=dict(range=[50, 150], autorange=False),
                aspectratio_x=1,
                aspectratio_y=1,
                aspectratio_z=1,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {"frame": {"duration": 10}, "fromcurrent": True,},
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {"frame": {"duration": 0}, "mode": "immediate",},
                            ],
                        ),
                    ],
                )
            ],
        ),
        frames=[go.Frame(data=frame) for frame in frames],
    )

    return body_animation
