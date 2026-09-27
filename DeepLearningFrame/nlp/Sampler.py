import numpy as np
from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(
        self,
        target: np.ndarray,
        sample_size: int
    ) -> np.ndarray:
        """
        return negative samples
        """
        pass

class UnigramSampler(Sampler):

    def __init__(self,word_count:dict[int,int],power:float=0.75,rng:np.random.Generator=None) -> None:
        super().__init__()
        self.word_count=word_count
        self.power = power
        self.rng=np.random.default_rng() if rng is None else rng
        self.word_size=len(word_count)
        #convert dict to numpy data_structure
        counts=np.zeros(self.word_size)
        for word_id,count in word_count.items():
            counts[word_id]=count
        probabilities=counts**self.power
        self.probabilities=probabilities/np.sum(probabilities)


    def sample(self,target:np.ndarray,sample_size:int) -> np.ndarray:
        self.target = target
        batch_size=target.shape[0]
        samples=np.zeros((batch_size,sample_size),dtype=np.int32)
        for i in range(batch_size):
            current_target=self.target[i]
            while True:
                negative=self.rng.choice(self.word_size,size=sample_size,replace=False,p=self.probabilities)
                if current_target not in negative:
                    break
            samples[i]=negative
        return samples
