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

## 5. General Kubernetes Commands

**To see all currently running pods (to check their status):**
```bash
kubectl get pods
```

**To see all active jobs:**
```bash
kubectl get jobs
```
