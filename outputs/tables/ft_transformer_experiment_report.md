# FT-Transformer Experiment Report

Best run by validation classifier Rank IC: `ft_small_seed362559` (0.093598).
Official FT-Transformer run: `ft_small_seed362559` (validation classifier Rank IC=0.093598).

## Epoch 1 Check
- `ft_base_seed362559`: yes (best_epoch=1).
- `ft_lr2e5_seed362559`: no (best_epoch=4).
- `ft_lr5e5_seed362559`: yes (best_epoch=1).
- `ft_reg_seed362559`: yes (best_epoch=1).
- `ft_small_lr5e5_seed362559`: no (best_epoch=3).
- `ft_small_seed362559`: no (best_epoch=2).

## Findings
- Lower learning rates improved validation classifier Rank IC: no (`ft_lr5e5_seed362559`: delta=-0.001655; `ft_lr2e5_seed362559`: delta=-0.005447).
- Stronger regularization improved validation classifier Rank IC: yes (delta=0.002152).
- Smaller architecture was competitive: yes (best small-run delta=0.015270).
- Seed robustness runs are not part of the active grid; all active experiments use seed 362559.

Test metrics are reported for final comparison only and are not used for checkpoint selection.
