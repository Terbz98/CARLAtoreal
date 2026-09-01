#!/bin/bash
# Paths this pipeline needs. Override any of them in the environment; the defaults assume the
# layout described in README.md.
#
# CARLA2REAL_ROOT locates itself from this file, so a clone works wherever it is put.
export CARLA2REAL_ROOT="${CARLA2REAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Where recordings, generated conditioning channels and training corpora live. This is bulk data
# (hundreds of GB) and is deliberately outside the repository.
export CARLA2REAL_DATA="${CARLA2REAL_DATA:-$CARLA2REAL_ROOT/datasets}"

# Where finished clips are written.
export CARLA2REAL_OUT="${CARLA2REAL_OUT:-$CARLA2REAL_ROOT/output}"

# OPTIONAL: an external perception stack used only for scoring (score_vp.py). Leave unset to skip
# scoring entirely -- nothing in the render or delivery path depends on it.
export PERCEPTION_ROOT="${PERCEPTION_ROOT:-}"
export PERCEPTION_DATA="${PERCEPTION_DATA:-$CARLA2REAL_OUT}"

# CARLA simulator install, needed only for recording new drives.
export CARLA_ROOT="${CARLA_ROOT:-$HOME/carla/simulator/CARLA_0.9.16}"

# Conda environment used for every python entry point.
export CARLA2REAL_ENV="${CARLA2REAL_ENV:-carla_env}"
