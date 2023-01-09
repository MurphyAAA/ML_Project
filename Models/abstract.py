from abc import ABC, abstractmethod

class Models (ABC):
    @abstractmethod
    def train(self):
        pass
    def validation(self):
        pass