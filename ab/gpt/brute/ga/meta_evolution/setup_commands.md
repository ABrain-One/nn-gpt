# Meta-Evolution Job Commands

This file contains the common commands needed to run the Meta-Evolution and Meta-Baseline jobs, as well as how to view their live logs.

## 1. Meta-Evolution CIFAR-10 (LLM-Guided GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar10

**To run the Mistral and DeepSeek variants (CIFAR-10):**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10_mistral.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10_mistral.json

kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10_deepseek.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10_deepseek.json
```

**To view live logs for Mistral and DeepSeek variants:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar10-mistral
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar10-deepseek
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar10.json
```

**To delete existing .pkl files (force fresh restart of GA population):**
```bash
rm -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar10_pipeline/GenFractal_ckpt_cifar10.pkl
```

**To delete existing LLM fine-tuned weights (force fresh restart of LLM's memory):**
```bash
rm -rf /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar10_pipeline/*_adapter
```

---

## 2. Meta-Baseline CIFAR-10 (Standard GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar10.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar10.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-baseline-ga-benchmark-cifar10
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar10.json
```

**To delete existing .pkl files:**
```bash
rm -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar10_pipeline/fractal_baseline_save_point_cifar10.pkl
```

---

## 3. Meta-Evolution CIFAR-100 (LLM-Guided GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar100

**To run the Mistral and DeepSeek variants (CIFAR-100):**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100_mistral.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100_mistral.json

kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100_deepseek.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100_deepseek.json
```

**To view live logs for Mistral and DeepSeek variants:**
```bash
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar100-mistral
kubectl logs -f job/nngpt-fractal-meta-evo-clonescience-cifar100-deepseek
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/meta_evol_tune_nngpt_cifar100.json
```

**To delete existing .pkl files (force fresh restart of GA population):**
```bash
rm -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/GenFractal_ckpt_cifar100.pkl
```

**To delete existing LLM fine-tuned weights (force fresh restart of LLM's memory):**
```bash
rm -rf /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/*_adapter
```

---

## 4. Meta-Baseline CIFAR-100 (Standard GA)

**To restart or run the job freshly:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar100.json --ignore-not-found=true && kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar100.json
```

**To view the live logs of the running job:**
```bash
kubectl logs -f job/nngpt-baseline-ga-benchmark-cifar100
```

**To stop and delete the job:**
```bash
kubectl delete -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/base_evol_tune_nngpt_cifar100.json
```

**To delete existing .pkl files:**
```bash
rm -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/cifar100_pipeline/fractal_baseline_save_point_cifar100.pkl
```

---

## 4.5 Frozen Control Experiment (No LLM)

These jobs run the meta-evolution schedule using only the handcrafted mutation operators (the LLM is bypassed using an identity step). These are used to isolate the impact of the meta-evolution schedule vs the LLM rewrites.

**To run the 1-iteration smoke test on CIFAR-100:**
```bash
kubectl apply -f /shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution/frozen_cifar100_test.json
```

**To launch the FINAL 5-iteration experiments on CIFAR-100:**
```bash
kubectl delete job --all
kubectl apply -f frozen_cifar100_run1.json -f frozen_cifar100_run2.json -f frozen_cifar100_run3.json -f frozen_cifar100_run4.json -f frozen_cifar100_run5.json
```

**To launch the FINAL 5-iteration experiments on CIFAR-10:**
```bash
kubectl delete job --all
kubectl apply -f frozen_cifar10_run1.json -f frozen_cifar10_run2.json -f frozen_cifar10_run3.json -f frozen_cifar10_run4.json -f frozen_cifar10_run5.json
```

**To view the live logs of the running Frozen jobs:**
```bash
# CIFAR-100 Logs
kubectl logs -f job/nngpt-fractal-frozen-cifar100-test
kubectl logs -f job/nngpt-fractal-frozen-cifar100-run1
kubectl logs -f job/nngpt-fractal-frozen-cifar100-run2
kubectl logs -f job/nngpt-fractal-frozen-cifar100-run3

# CIFAR-10 Logs
kubectl logs -f job/nngpt-fractal-frozen-cifar10-run1
kubectl logs -f job/nngpt-fractal-frozen-cifar10-run2
kubectl logs -f job/nngpt-fractal-frozen-cifar10-run3
```

**To preview the live logs of the running Frozen CIFAR-100 jobs without leaving the terminal:**
```bash
python3 -c "
import os
base_dir = '/shared/ssd/home/b-a-singh/Thesis/cloneScience/nn-gpt/ab/gpt/brute/ga/meta_evolution'
cifar100_logs = os.path.join(base_dir, 'cifar100_pipeline', 'logs_cifar100', 'frozen')
if not os.path.exists(cifar100_logs): print('No frozen logs found yet.')
else:
    runs = os.listdir(cifar100_logs)
    print(f'Runs found: {len(runs)}')
    for run in runs:
        run_path = os.path.join(cifar100_logs, run)
        files = os.listdir(run_path)
        e1 = [f for f in files if f.endswith('_1_epoch.jsonl')]
        e3 = [f for f in files if f.endswith('_true.jsonl')]
        pr = [f for f in files if f.endswith('.jsonl') and not f.endswith('true.jsonl') and not f.endswith('epoch.jsonl') and not f.startswith('frozen_meta_log')]
        print(f'\nRun [{run}]:\n  _1_epoch files: {len(e1)}\n  _true files: {len(e3)}\n  predictor files: {len(pr)}')
        if pr:
            with open(os.path.join(run_path, pr[0])) as f:
                print(f'  -> Predictor evaluations logged: {len(f.readlines())}')
"
```

---

## 5. General Kubernetes Commands

**To see all currently running pods (to check their status):**
```bash
kubectl get pods
```

**To see all active jobs:**
```bash
kubectl get jobs
```

---

## 6. Unified Start Command

**To restart or run ALL 8 jobs freshly at once:**
```bash
kubectl delete -f base_evol_tune_nngpt_cifar10.json --ignore-not-found=true && kubectl apply -f base_evol_tune_nngpt_cifar10.json && \
kubectl delete -f base_evol_tune_nngpt_cifar100.json --ignore-not-found=true && kubectl apply -f base_evol_tune_nngpt_cifar100.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar10.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar10.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar10_mistral.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar10_mistral.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar10_deepseek.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar10_deepseek.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar100.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar100.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar100_mistral.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar100_mistral.json && \
kubectl delete -f meta_evol_tune_nngpt_cifar100_deepseek.json --ignore-not-found=true && kubectl apply -f meta_evol_tune_nngpt_cifar100_deepseek.json
```
