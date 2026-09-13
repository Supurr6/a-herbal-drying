# -*- coding: utf-8 -*-
"""问题四固定参考域下的独立累计水分收支核验。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "q2_full"))
sys.path.insert(0, str(ROOT / "src" / "q4_shrinkage"))
import solver_q2 as q2
import solver_q4 as q4

def audit(path: Path):
    data = np.load(path)
    t, c, xi, radius = data["t"], data["C"], data["xi"], data["R"]
    inventory = np.trapezoid(c * xi, xi, axis=1)
    ambient = q4.ambient_moisture(t)
    outward_flux = q2.H_M / radius * (c[:, -1] - ambient)
    discharged = np.zeros_like(t)
    discharged[1:] = np.cumsum(0.5 * (outward_flux[1:] + outward_flux[:-1]) * np.diff(t))
    residual = inventory - inventory[0] + discharged
    loss = inventory[0] - inventory[-1]
    return {
        "coordinate": "xi=r/R(t) fixed reference domain",
        "initial_reference_inventory": float(inventory[0]),
        "total_reference_loss": float(loss),
        "final_relative_residual": float(abs(residual[-1]) / loss),
        "maximum_relative_residual": float(np.max(np.abs(residual)) / loss),
    }

if __name__ == "__main__":
    path = ROOT / "outputs" / "q4" / "q4_4hfixed_dt5_k1.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        result = q4.run(refine=1, dt=5.0)
        np.savez(path, **result)
    result = audit(path)
    out = ROOT / "outputs" / "q4" / "q4_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
