import plotly.graph_objects as go
import plotly.io as pio

pio.templates["ds_econ"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="ds_econ watermark",
            text="DS-Econ",
            textangle=-30,
            opacity=0.05,
            font=dict(color="black", size=100),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    ]
)
