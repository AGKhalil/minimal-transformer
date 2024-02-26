import csv
import os
import re

import numpy as np
from tqdm import tqdm


def download_and_unzip_data(data_path: str = ".data") -> None:
    """
    Downloads and unzips a dataset from
    'https://www.manythings.org/anki/deu-eng.zip',placing the
    contents in the specified data path.

    Args:
        data_path (str): The target directory where the downloaded
                         zip file will be placed. Defaults to
                         '.data' if not specified.
    """
    os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    os.system("wget https://www.manythings.org/anki/deu-eng.zip")
    os.system("unzip deu-eng.zip")
    os.remove("deu-eng.zip")
    os.chdir("..")


def create_tokenized_dataset(
    data_path: str = ".data",
    dataset_size: int = 10000,
    sentence_length: int = 10,
) -> None:
    """Creates a tokenized dataset from the given text file containing English-
    German sentence pairs. The sentences are tokenized, truncated or padded to
    a fixed length, converted to lowercase, and saved into separate CSV files
    for English and German.

    Args:
        data_path (str): The directory containing the input text file with
                         sentence pairs. Defaults to '.data' if not specified.
        dataset_size (int): Number of random sentence pairs to extract from
                            the raw dataset. Defaults to 10000 if not specified.
        sentence_length (int): Fixed length that each sentence will be padded
                               or truncated to. Defaults to 10 if not specified.
    """
    with open(os.path.join(data_path, "deu.txt")) as f:
        sentences = f.readlines()

    eng_sentences, deu_sentences = [], []
    eng_words, deu_words = set(), set()
    for i in tqdm(range(dataset_size)):
        rand_idx = np.random.randint(len(sentences))
        # find only letters in sentences
        eng_sent, deu_sent = ["<sos>"], ["<sos>"]
        eng_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[0])
        deu_sent += re.findall(r"\w+", sentences[rand_idx].split("\t")[1])

        # change to lowercase
        eng_sent = [x.lower() for x in eng_sent]
        deu_sent = [x.lower() for x in deu_sent]
        eng_sent.append("<eos>")
        deu_sent.append("<eos>")

        if len(eng_sent) >= sentence_length:
            eng_sent = eng_sent[:sentence_length]
        else:
            for _ in range(sentence_length - len(eng_sent)):
                eng_sent.append("<pad>")

        if len(deu_sent) >= sentence_length:
            deu_sent = deu_sent[:sentence_length]
        else:
            for _ in range(sentence_length - len(deu_sent)):
                deu_sent.append("<pad>")

        # add parsed sentences
        eng_sentences.append(eng_sent)
        deu_sentences.append(deu_sent)

        # update unique words
        eng_words.update(eng_sent)
        deu_words.update(deu_sent)

    eng_words, deu_words = list(eng_words), list(deu_words)

    # encode each token into index
    for i in tqdm(range(len(eng_sentences))):
        eng_sentences[i] = [eng_words.index(x) for x in eng_sentences[i]]
        deu_sentences[i] = [deu_words.index(x) for x in deu_sentences[i]]

    # save english and deutch csvs
    eng_csv = os.path.join(data_path, "eng.csv")
    with open(eng_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(eng_sentences)

    deu_csv = os.path.join(data_path, "deu.csv")
    with open(deu_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(deu_sentences)
