"""Compatibility shim for the local web dashboard app."""

from ..horizon_ext.web.app import *  # noqa: F401,F403
from ..horizon_ext.web.app import _run_pipeline, _stage_payloads  # noqa: F401
