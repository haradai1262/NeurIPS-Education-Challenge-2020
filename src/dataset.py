
import torch


class SimpleDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert all([dataset[i].size(0) == dataset[0].size(0) for i in range(len(dataset))]), 'all the elemtnes must have the same length'
        self.data_size = dataset[0].size(0)

    def __iter__(self):
        self._i = 0

        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.dataset = [v[index_shuffle] for v in self.dataset]

        return self

    def __next__(self):

        i1 = self.batch_size * self._i
        i2 = min(self.batch_size * (self._i + 1), self.data_size)

        if i1 >= self.data_size:
            raise StopIteration()

        value = [v[i1:i2] for v in self.dataset]
        self._i += 1
        return value
