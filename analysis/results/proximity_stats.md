# Time spent close to the caregiver

Position source: Gazebo ground truth. Close means inside the square around the caregiver whose side is half the arena width. Stuck periods are excluded. Every 10 Hz sample counts as one observation, as in the paper, so the p-values overstate the evidence from a single run per regime.

| Regime | Run | Samples | Close to caregiver |
|---|---|---|---|
| Secure (avoidance 0, ambivalence 0.1) | experiment_data_34 | 2764 | 56.7% |
| Strongly avoidant (avoidance 0.9, ambivalence 0; oscillator model diverged) | experiment_data_35 | 3009 | 96.9% |
| Moderately ambivalent (avoidance 0, ambivalence 0.4) | experiment_data_36 | 2856 | 62.6% |

Chi-squared = 1392.11, df = 2, N = 8629, p = 5.1e-303

| Pair | z | p (Bonferroni) | Differ at 0.05 |
|---|---|---|---|
| Secure vs Strongly avoidant | -36.66 | 9.02e-294 | yes |
| Secure vs Moderately ambivalent | -4.52 | 1.88e-05 | yes |
| Strongly avoidant vs Moderately ambivalent | 32.97 | 5.61e-238 | yes |
