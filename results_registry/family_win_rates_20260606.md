# Target-Family Win Rates

Source: `/home/kabir/newws/acc_gap_advantage_analysis.json` mined from existing summary/meta JSON logs. No experiments run.

| target_family | n | wins | ties | losses | win_rate | non_loss_rate | mean_advantage | median_advantage | mean_acc_gap | datasets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alt_nn4 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | +0.3291 | +0.3291 | 0.5843 | cifar-10 |
| alt_nn2 | 1 | 1 | 0 | 0 | 100.0% | 100.0% | +0.0057 | +0.0057 | 0.6468 | cifar-10 |
| ga | 1 | 1 | 0 | 0 | 100.0% | 100.0% | +0.0042 | +0.0042 | 0.7408 | cifar-10 |
| alt_nn1 | 10 | 8 | 0 | 2 | 80.0% | 80.0% | +0.1314 | +0.1837 | 0.6530 | cifar-10 |
| alt | 5 | 4 | 0 | 1 | 80.0% | 80.0% | +0.0150 | +0.0211 | 0.8255 | cifar-10,mnist |
| alexnet | 9 | 7 | 0 | 2 | 77.8% | 77.8% | +0.0814 | +0.0181 | 0.6216 | celeba-gender,cifar-100,imagenette,svhn |
| bagnet | 12 | 6 | 0 | 6 | 50.0% | 50.0% | -0.0038 | +0.0012 | 0.5748 | cifar-100,imagenette,svhn |
| shufflenet | 4 | 2 | 0 | 2 | 50.0% | 50.0% | +0.0063 | +0.0086 | 0.5560 | cifar-100 |
| dpn68 | 4 | 2 | 0 | 2 | 50.0% | 50.0% | -0.0060 | +0.0090 | 0.7658 | svhn |
| airnext | 8 | 3 | 1 | 4 | 37.5% | 50.0% | +0.0138 | +0.0000 | 0.6175 | imagenette,svhn |
| darknet | 8 | 3 | 3 | 2 | 37.5% | 75.0% | +0.0083 | +0.0000 | 0.4681 | cifar-100,imagenette |
| alt_nn3 | 1 | 0 | 0 | 1 | 0.0% | 0.0% | -0.0215 | -0.0215 | 0.6035 | cifar-10 |
