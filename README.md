# EP60 — Monitoring Deep Learning Models in Production

**Drift · Degradation · Alerting · Retraining Trigger**

Part of the CodeToAGI Deep Learning Series (72 episodes · Neurons → GPT)

## What this episode does

We deploy a simple digit classifier and send it **8 weeks of simulated live traffic** under two failure modes:

1. **Sensor / input drift** – camera noise grows every week  
2. **Concept drift** – labels of classes 3 and 8 swap from week 4 onward

We watch three signals:
- Share of drifted pixel columns (Kolmogorov-Smirnov + optional Evidently)
- Mean prediction confidence
- Accuracy (the ground truth, but labels arrive late)

We also measure false-alarm rates on clean traffic and fire a retraining trigger when accuracy falls below 90% (same gate as EP59).

## Quick start

```bash
pip install torch scikit-learn numpy pandas scipy matplotlib "evidently>=0.7"
python ep60_monitor.py
