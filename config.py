"""Paths this pipeline needs, resolved from the environment with repository-relative defaults.

Mirrors config.sh so the shell drivers and the python tools agree on where things are. Import it
instead of hardcoding a path:

    from config import ROOT, DATA, OUT
"""
import os

ROOT = os.environ.get('CARLA2REAL_ROOT') or os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('CARLA2REAL_DATA') or os.path.join(ROOT, 'datasets')
OUT = os.environ.get('CARLA2REAL_OUT') or os.path.join(ROOT, 'output')
RESULTS = os.path.join(ROOT, 'pix2pixHD', 'results')
CHECKPOINTS = os.path.join(ROOT, 'pix2pixHD', 'checkpoints')

# Optional external perception stack, used only for scoring.
PERCEPTION_ROOT = os.environ.get('PERCEPTION_ROOT') or ''
PERCEPTION_DATA = os.environ.get('PERCEPTION_DATA') or OUT
