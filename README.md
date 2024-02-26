# minimal-transformer
This repo contains a minimal seq2seq transformer implementation and is applied to a subset of the english-deutch translation task.

# Installation
```
git clone 
cd minimal-transformer
pip install -e .
```

# Usage
```
python train.py
```

# Note
The dataset is tokenized and padded to length `N` prior to training, this means the training loop itself does not rely on a tokenizer, nor a collator for padding sequences of different lengths. 