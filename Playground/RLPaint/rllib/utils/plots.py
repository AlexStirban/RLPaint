"""Contains auxiliary functions used for RL plots."""

# Partial imports
from typing import Sequence

# Aliased imports
import pandas as pd
import plotly.graph_objects as go

def make_live_plot(var_label: str = ""):
    """Plots the rewards using a Plotly lineplot while applying a
       a rolling mean.

    Args:
        rewards (Sequence[float]): Rewards sequence
        roll_size (int, optional): Number of instances to use for the rolling mean. 
        Defaults to 100.
    """


    # Plot
    fig = go.FigureWidget()

    # Add traces
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="lines", name=var_label)
    )

    fig.add_trace(
        go.Scatter(
            x=[None], 
            y=[None],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode="lines", 
            name="Roll. Avg. Reward",
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            fillcolor='rgba(68, 68, 68, 0.3)',
            marker=dict(color="#444"),
            line=dict(width=0),
            fill='tonexty',
            mode="lines",
            name="Roll. Avg. Reward",
            showlegend=False
        )
    )


    # Update fig optiosn
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Episode",
        yaxis_title=var_label
    )

    return fig


def update_live_plot(fig: go.FigureWidget, var: Sequence[float], roll_size: int=50):
    # Make pd
    data_pd = pd.DataFrame({"v": var})

    # Compute confidence intervals
    data_pd = (data_pd
        .assign(roll_avg=lambda x: x.v.rolling(roll_size).mean())
        .assign(roll_std=lambda x: x.v.rolling(roll_size).std())
        .assign(upper=lambda x: x.roll_avg + 1.96 * x.roll_std)
        .assign(lower=lambda x: x.roll_avg - 1.96 * x.roll_std)
    )

    # Update avg
    line_data = fig.data[0]
    line_data.x = data_pd.index
    line_data.y = data_pd.roll_avg

    # Update lowe bound
    upper_data = fig.data[1]
    upper_data.x = data_pd.index
    upper_data.y = data_pd.upper
    
    # Update lowe bound
    lower_data = fig.data[2]
    lower_data.x = data_pd.index
    lower_data.y = data_pd.lower 

