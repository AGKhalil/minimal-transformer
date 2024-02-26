# minimal-transformer
This repo contains a minimal seq2seq transformer implementation and is applied to a subset of the english-deutch translation task.

# Installation
```
conda create -n mint python==3.8.1
conda activate mint

git clone git@github.com:AGKhalil/minimal-transformer.git
cd minimal-transformer
pip install -e .
```

# Usage
```
python train.py
```

# Note
The dataset is tokenized and padded to length `N` prior to training, this means the training loop itself does not rely on a tokenizer, nor a collator for padding sequences of different lengths. 