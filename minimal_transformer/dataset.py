import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MTDataset(Dataset):
    """A dataset class representing a machine translation dataset where each
    entry consists of a pair of sentences, one in English and the other in
    German.

    Attributes:
        source (np.ndarray): Array containing the English sentences.
        target (np.ndarray): Array containing the corresponding German
            sentences.
    """

    def __init__(
        self,
        eng_csv_path: str = ".data/eng.csv",
        deu_csv_path: str = ".data/deu.csv",
    ) -> None:
        """Initializes the MTDataset with data from CSV files containing
        English and German sentences.

        Parameters:
            eng_csv_path (str): The path to the CSV file containing English
                sentences.
            deu_csv_path (str): The path to the CSV file containing German
                sentences.
        """
        eng_sentences = pd.read_csv(eng_csv_path)
        deu_sentences = pd.read_csv(deu_csv_path)

        self.source = np.array(eng_sentences, dtype=int)
        self.target = np.array(deu_sentences, dtype=int)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieves the source-target pair at the specified index.

        Parameters:
            idx (int): The index of the desired data point.

        Returns:
            tuple: A tuple containing the source sentence at index `idx` and its
                corresponding target sentence.
        """
        return self.source[idx], self.target[idx]

    def __len__(self) -> int:
        """Returns the length of the dataset, i.e., the number of source-target
        pairs.

        Returns:
            int: The total number of entries in the dataset.
        """
        return len(self.source)
