from abc import ABC, abstractmethod

class Quantizer(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def quantize(self, x):
        pass