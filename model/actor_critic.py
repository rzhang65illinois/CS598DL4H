import torch
import torch.nn as nn
import torch.nn.functional as F
class Actor(nn.Module):
    def __init__(self, max_length):
        super(Actor, self).__init__()
        self.max_length = max_length 
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, self.max_length)

    def forward(self, x):
        samples, entropy, logprob = [], [], []
        scores = self.fc(self.encoder(x))
        for i in range(self.max_length):
            odds = torch.exp(scores[:,i,:])
            prob = odds / (1.0 + odds)
            prob_dist = torch.distributions.bernoulli.Bernoulli(prob)
            s = prob_dist.sample()
            e = prob_dist.entropy()
            l = prob_dist.log_prob(s)
            samples.append(s)
            entropy.append(e)
            logprob.append(l)
        graphs = torch.permute(torch.stack(samples), (1,0,2))
        return graphs, scores, entropy, logprob

class Critic(nn.Module):
    def __init__(self, max_length):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(max_length**2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        return v
