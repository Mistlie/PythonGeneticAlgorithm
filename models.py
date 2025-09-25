import torch
import torch.nn as nn

"""Оценка качества расписания"""
class QualityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

"""Оптимизация параметров генетического алгоритма"""
class GAOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 3), nn.Softmax(dim=1)
        )
    def forward(self, x): return self.network(x)

"""Нейросеть для оценки и оптимизации"""
class ScheduleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
        )
        self.quality_predictor = QualityPredictor(hidden_size, hidden_size // 2)
        self.ga_optimizer = GAOptimizer(hidden_size, hidden_size // 2)
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.quality_predictor(features), self.ga_optimizer(features)