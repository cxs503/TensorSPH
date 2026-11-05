import sys, math, json, time
sys.path.insert(0,'src')
from tensorsph.dam_break import DamBreakConfig, run_case, ritter_front
import torch
dx=0.04
cfg=DamBreakConfig(dx=dx, steps=int(round(0.5/cfg_dt)) if False else 800002)
