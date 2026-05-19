# Minimum Sufficient Result Package

If time is short, the results chapter should include exactly this package:

1. **Figure 1:** cross-evaluation success-rate heatmap.
2. **Figure 2:** retention-drop bar chart comparing the MTL policy against the specialist policy for each task.
3. **Figure 3:** phase/checkpoint curves for A1, A2, B1, B2, and C2.
4. **Table 1:** family-level averages for Family A, Family B, and Family C.
5. **Appendix:** failure-rate and mean-alive-time matrices.

This is sufficient to support both research questions. It shows not only that the policy performs, but also where transfer happens, where interference appears, and whether the task design reveals meaningful structure.

Generate the package with:

```powershell
C:\Users\pavel\anaconda3\envs\env_isaaclab\python.exe analysis\generate_result_package.py
```

The script writes its outputs to:

```text
results/figures/result_package/
```

If only the cross-evaluation matrix needs to be regenerated, use:

```powershell
C:\Users\pavel\anaconda3\envs\env_isaaclab\python.exe analysis\export_eval_matrix.py --results_root results --metric success_rate --exclude MTL_unified
```
