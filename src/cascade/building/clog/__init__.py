"""Pipe-clog testing and prevention ("Clog Watch", module key ``clog``).

Advisory only. Cerebro proposes tests, alarms and a check schedule; a person approves every action. The nightly
active flow test runs on DOMESTIC COLD WATER ONLY through a motorised test-draw valve that the BMS operates on a
human-approved standing schedule (with a BMS-side low-pressure abort). Nothing here touches fire, sprinkler or
standpipe piping.

Modules (runtime-safe ones import only numpy/pandas/scikit-learn):
  rules      rule and threshold table (rules.json) with PUBLIC / team-proposed / Assumption tags and URLs
  geometry   code-plausible 32-floor riser design (3 pressure zones, every outlet <= 80 psi static)
  demand     REAL HSB Living Lab apartment demand -> floors x 10-min steps
  sim_riser  nightly test schedule for the riser simulation (the WNTR code is in scripts/clog_sim_lib.py)
  features   nightly active-test and passive features, noise model
  detect     z-score rule, gradient-boosting models, out-of-fold thresholds, scenario bootstrap
  drain      drain-down ratio rule (the pyswmm model is in scripts/clog_sim_lib.py)
  bellinge   REAL Bellinge July 2020 blockage detectors (paired rule, fixed level, residual)
  schedule   risk-based periodic-check scheduler
  svg        riser-and-stack sensor diagram
"""
