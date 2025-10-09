"""CLI entry-point for the RCC dashboard suite."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..backend import LawCheckpoint, load_checkpoint
from ..style import get_theme
from .echo_playback import build_echo_frames, make_echo_animation
from .gamma_loop import compute_gamma_loops, make_gamma_figure
from .neuro_vis import build_neuro_semantic_trace, make_neuro_figure
from .torsion_map import compute_torsion_map, make_torsion_heatmap

try:  # pragma: no cover - optional dependency boundary
    from dash import Dash, Input, Output, dcc, html
except Exception:  # pragma: no cover - fallback when dash not installed
    Dash = None  # type: ignore
    Input = Output = None  # type: ignore
    dcc = html = None  # type: ignore


@dataclass
class DashboardConfig:
    """Configuration payload for dashboard creation."""

    checkpoint: LawCheckpoint
    style: str
    enable_echo: bool


DEFAULT_PHI = float(np.pi / 6)
DEFAULT_Y0 = 0.5
DEFAULT_CHI = 1.0


def _build_layout(app: Dash, config: DashboardConfig):  # pragma: no cover - layout config
    theme = get_theme(config.style)
    gamma_result = compute_gamma_loops(DEFAULT_Y0, DEFAULT_PHI)
    torsion_result = compute_torsion_map(DEFAULT_CHI)
    law_tokens = config.checkpoint.metadata.get("law_tokens", [])
    neuro_trace = build_neuro_semantic_trace(law_tokens)

    graphs = [
        html.Div(
            [
                html.H3("γ(r, θ) Law Sculptor"),
                dcc.Graph(id="gamma-graph", figure=make_gamma_figure(gamma_result)),
                html.Div(id="gamma-loop-area", children=f"Loop area: {gamma_result.loop_area:.3f}"),
                dcc.Slider(
                    id="phi-slider",
                    min=0.0,
                    max=float(np.pi / 2),
                    step=0.01,
                    value=DEFAULT_PHI,
                    marks={0: "0", float(np.pi / 2): "π/2"},
                ),
                dcc.Slider(
                    id="y0-slider",
                    min=0.2,
                    max=0.8,
                    step=0.01,
                    value=DEFAULT_Y0,
                    marks={0.2: "0.2", 0.8: "0.8"},
                ),
                html.Button("Export γ-loop", id="export-gamma", n_clicks=0),
            ],
            className="panel",
        ),
        html.Div(
            [
                html.H3("Torsion Norm Explorer"),
                dcc.Dropdown(
                    id="chi-dropdown",
                    options=[
                        {"label": f"χ intensity {scale:.1f}", "value": scale}
                        for scale in (0.5, 1.0, 1.5)
                    ],
                    value=DEFAULT_CHI,
                ),
                dcc.Checklist(
                    id="golden-perturb",
                    options=[{"label": "Enable golden perturbation", "value": "golden"}],
                    value=[],
                ),
                dcc.Graph(
                    id="torsion-graph",
                    figure=make_torsion_heatmap(torsion_result),
                ),
            ],
            className="panel",
        ),
        html.Div(
            [
                html.H3("NeuroSemantic Coupling"),
                dcc.Graph(id="neuro-graph", figure=make_neuro_figure(neuro_trace)),
            ],
            className="panel",
        ),
    ]

    if config.enable_echo:
        echo = build_echo_frames(config.checkpoint.law_tensor)
        graphs.append(
            html.Div(
                [
                    html.H3("Temporal Echo Playback"),
                    dcc.Graph(id="echo-graph", figure=make_echo_animation(echo)),
                ],
                className="panel",
            )
        )

    app.layout = html.Div(
        graphs,
        style={
            "backgroundColor": theme.background,
            "color": theme.foreground,
            "padding": "1rem",
            "fontFamily": "'IBM Plex Sans', sans-serif",
        },
    )


def _register_callbacks(app: Dash) -> None:  # pragma: no cover - callback wiring
    @app.callback(
        Output("gamma-graph", "figure"),
        Output("gamma-loop-area", "children"),
        Input("phi-slider", "value"),
        Input("y0-slider", "value"),
    )
    def _update_gamma(phi: float, y0: float):
        result = compute_gamma_loops(y0, phi)
        figure = make_gamma_figure(result)
        summary = f"Loop area: {result.loop_area:.3f}"
        return figure, summary

    @app.callback(
        Output("torsion-graph", "figure"),
        Input("chi-dropdown", "value"),
        Input("golden-perturb", "value"),
    )
    def _update_torsion(chi_value: float, perturb_flags: list[str]):
        perturb = 0.0
        if perturb_flags:
            perturb = 1.0
        result = compute_torsion_map(chi_value, perturb=perturb)
        return make_torsion_heatmap(result)


def create_dashboard_app(config: DashboardConfig) -> Dash:
    """Create the Dash app using *config*."""

    if Dash is None:
        raise RuntimeError(
            "Dash is not installed. Install 'dash' to launch the dashboard interface."
        )

    app = Dash(__name__)
    _build_layout(app, config)
    _register_callbacks(app)
    return app


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Launch RCC law dashboards")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--port", type=int, default=8052, help="Port to bind server")
    parser.add_argument("--style", default="QuantumDark", help="Dashboard theme name")
    parser.add_argument(
        "--echo-playback", action="store_true", help="Enable temporal echo playback pane"
    )
    parser.add_argument(
        "--enable-live-feed",
        action="store_true",
        help="Reserved flag enabling live RCC feed integration",
    )
    parser.add_argument("--debug", action="store_true", help="Run server in debug mode")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry-point for running the dashboard from the command line."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    checkpoint = load_checkpoint(args.checkpoint)
    config = DashboardConfig(
        checkpoint=checkpoint,
        style=args.style,
        enable_echo=bool(args.echo_playback),
    )

    if Dash is None:
        print(
            "Dash is not installed. Install 'dash' and 'plotly' to run the dashboard.\n"
            "Loaded checkpoint metadata:"
        )
        for key, value in checkpoint.metadata.items():
            print(f"  - {key}: {value}")
        return 0

    app = create_dashboard_app(config)
    app.run_server(host="0.0.0.0", port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
