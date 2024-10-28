# SkrullSeek

<img src="assets/logo.png" alt="Skrullseek logo" width="200"/>

Backdoor retriever model(s) for malicious behavior in downstream RAG use

# Setup

Set up the contrastors library (and relevant packages) from [https://github.com/iamgroot42/contrastors](https://github.com/iamgroot42/contrastors).

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.9.0 --no-deps
```

# Structure

- Backdoor_DPR: Testing out data-level poisoning on DPR models