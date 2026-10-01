#!/usr/bin/env python3
"""Print the input-channel count of a checkpoint's first conv.

Exists because the preflight in train_generic.sh has to know how wide a parent is BEFORE training.
Written as a FILE rather than a heredoc: `conda run` discards stdin, so the first version of this
check printed "?" for every run and silently told the preflight nothing.

  usage: read_conv_in.py <latest_net_G.pth>
"""
import sys

import torch

sd = torch.load(sys.argv[1], map_location='cpu')
for v in sd.values():
    if hasattr(v, 'dim') and v.dim() == 4:
        print(v.shape[1])
        break
