import random
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class EpochConcateSampler(Sampler):
    """Concatenate  all epoch index arrays into one index array.
    Arguments:
        data_source (Dataset): dataset to sample from
        epoch(int): epoch num
    """

    def __init__(self, data_source, epoch):
        self.data_length = len(data_source)
        self.epoch = epoch

    def __iter__(self):
        index_all = []
        for i in range(self.epoch):
            index = list(range(self.data_length))
            random.shuffle(index)
            index_all += index
        return iter(index_all)

    def __len__(self):
        return self.data_length * self.epoch


class EpochConcateSamplerResume(Sampler):
    """Concatenate  all epoch index arrays into one index array.
    Arguments:
        data_source (Dataset): dataset to sample from
        epoch(int): epoch num
        resume_epoch(int): at the begining for training, resume_epoch=0
    """

    def __init__(self, data_source, epoch, resume_epoch):
        self.data_length = len(data_source)
        self.epoch = epoch
        self.resume_epoch = resume_epoch

    def __iter__(self):
        index_all = []
        end_epoch = self.resume_epoch + self.epoch
        for i in range(self.resume_epoch, end_epoch):
            index = list(range(self.data_length))
            random.seed(i)
            random.shuffle(index)
            index_all += index
        return iter(index_all)

    def __len__(self):
        return self.data_length * self.epoch


class EpochConcateDistributedSampler(DistributedSampler):
    """Concatenate partition of all epoch index arrays into one index array for distributed.
        Arguments:
            data_source (Dataset): dataset to sample from
            epoch(int): epoch num
            resume_epoch(int): at the begining for training, resume_epoch=0
        """

    def __init__(self, data_source, epoch, resume_epoch=0):
        super(EpochConcateDistributedSampler, self).__init__(data_source)
        self.data_length = len(data_source)
        self.epoch = epoch
        self.resume_epoch = resume_epoch

    def __iter__(self):
        index_all = []
        end_epoch = self.resume_epoch + self.epoch
        for i in range(self.resume_epoch, end_epoch):
            super(EpochConcateDistributedSampler, self).set_epoch(i)
            index_all += list(super(EpochConcateDistributedSampler, self).__iter__())
        return iter(index_all)

    def __len__(self):
        return super(EpochConcateDistributedSampler, self).__len__() * self.epoch