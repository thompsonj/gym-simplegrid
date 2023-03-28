"""Pytorch neural networks for use in Actor-Critic RL"""
import torch
from torch import nn

class PolicyMLP(nn.Module):
    
    def __init__(self, n_actions, n_states, hidden_size):
        super().__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.n_states, self.hidden_size)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_activation = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.n_actions)
        self.pi = nn.Softmax(dim = 0)
    
    def forward(self, input):

        X = self.fc1(input)
        out = self.fc2(self.fc1_activation(X))
        out = self.fc3(self.fc2_activation(out))
        output = self.pi(out)
        return output

class CriticMLP(nn.Module):

    def __init__(self, n_states, hidden_size, n_outputs = 1):
        super().__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_states, self.hidden_size)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_activation = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.n_outputs)
    
    def forward(self, input):

        X = self.fc1(input)
        output = self.fc2(self.fc1_activation(X))
        output = self.fc3(self.fc2_activation(output))
        return output

class PolicyRNN(nn.Module):
    
    def __init__(self, n_actions, n_inputs, hidden_size=32):
        
        super().__init__()
        self.n_actions = n_actions
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        # torch.manual_seed(1)
        self.initialize_model()

    def initialize_model(self):
        self.emb  = nn.Linear(self.n_inputs, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.n_actions)
        self.logits = nn.LogSoftmax(dim=0)

    def forward(self, x, lstm_hidden = None):
        x = self.emb(x)
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)
        out = self.out(lstm_h)
        logits = self.logits(out)

        return logits, (lstm_h, lstm_c)

class CriticRNN(nn.Module):
    
    def __init__(self, n_inputs, hidden_size=32):
        
        super().__init__()
        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        torch.manual_seed(1)
        self.initialize_model()

    def initialize_model(self):
        self.emb  = nn.Linear(self.n_inputs, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1)

    def forward(self, x, lstm_hidden = None):
        x = self.emb(x)
        if lstm_hidden is None:
            lstm_h, lstm_c = self.lstm(x)
        else:
            lstm_h, lstm_c = self.lstm(x, lstm_hidden)
            
        out = self.out(lstm_h)

        return out, (lstm_h, lstm_c)