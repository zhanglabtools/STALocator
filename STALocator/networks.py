# Modified from Portal:
# Zhao J, et al. (2022) Adversarial domain translation networks for integrating large-scale atlas-level single-cell datasets. Nature Computational Science 2(5):317-330.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class encoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        z = F.linear(h, self.W_2, self.b_2)
        return z

class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x

class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)

class encoder_site(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder_site, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidd_1 = 1000
        n_hidd_2 = 500
        n_hidd_3 = 50
        n_hidd_4 = 10

        self.fc1 = nn.Linear(n_input, n_hidd_1)
        self.fc1_bn = nn.BatchNorm1d(n_hidd_1)
        self.fc2 = nn.Linear(n_hidd_1, n_hidd_2)
        self.fc2_bn = nn.BatchNorm1d(n_hidd_2)
        self.fc3 = nn.Linear(n_hidd_2, n_hidd_3)
        self.fc3_bn = nn.BatchNorm1d(n_hidd_3)
        self.fc4 = nn.Linear(n_hidd_3, n_hidd_4)
        self.fc4_bn = nn.BatchNorm1d(n_hidd_4)
        self.fc5 = nn.Linear(n_hidd_4, n_latent)

    def forward(self, input):
        h1 = F.relu(self.fc1_bn(self.fc1(input)))
        h2 = F.relu(self.fc2_bn(self.fc2(h1)))
        h3 = F.relu(self.fc3_bn(self.fc3(h2)))
        h4 = F.relu(self.fc4_bn(self.fc4(h3)))
        #return F.relu(self.fc5(h4))
        return self.fc5(h4)

class decoder_site(nn.Module):
    def __init__(self, n_input, n_latent):
        super(decoder_site, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidd_6 = 10
        n_hidd_7 = 50
        n_hidd_8 = 500
        n_hidd_9 = 1000

        self.fc6 = nn.Linear(n_latent, n_hidd_6)
        self.fc6_bn = nn.BatchNorm1d(n_hidd_6)
        self.fc7 = nn.Linear(n_hidd_6, n_hidd_7)
        self.fc7_bn = nn.BatchNorm1d(n_hidd_7)
        self.fc8 = nn.Linear(n_hidd_7, n_hidd_8)
        self.fc8_bn = nn.BatchNorm1d(n_hidd_8)
        self.fc9 = nn.Linear(n_hidd_8, n_hidd_9)
        self.fc9_bn = nn.BatchNorm1d(n_hidd_9)
        self.fc10 = nn.Linear(n_hidd_9, n_input)

    def forward(self, z):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        h9 = F.relu(self.fc9_bn(self.fc9(h8)))
        return self.fc10(h9)