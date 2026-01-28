"""Toolkit for RCC law dashboards and GPT operator control."""

from importlib import metadata


def get_version() -> str:
    """Return the package version if installed, otherwise ``'0.0.0'``."""
    try:
        return metadata.version("law_tools")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for source usage
        return "0.0.0"


__all__ = ["get_version"]
