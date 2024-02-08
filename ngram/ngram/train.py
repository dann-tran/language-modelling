from tqdm import tqdm

from ngram.dataloader import DataLoader
from ngram.model import NGramModel


class Trainer:
    def __init__(
        self,
        model: NGramModel,
        dataloader: DataLoader,
    ):
        self.model = model
        self.dataloader = dataloader

    def train(self):
        for docs in tqdm(self.dataloader):
            self.model.fit(docs)
