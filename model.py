## from PMI-SYSU

from src.utils import *

class SeparateSimpLinearAttn(nn.Module):
    def __init__(self, dim_embedding, d_k=None):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.d_k = d_k if d_k is not None else dim_embedding

        self.W_q = nn.Parameter(torch.normal(0, 1/np.sqrt(dim_embedding), size=(self.d_k, dim_embedding)))
        self.W_k = nn.Parameter(torch.normal(0, 1/np.sqrt(dim_embedding), size=(self.d_k, dim_embedding)))

    def forward(self, x):
        N = x.shape[-1]
        Q = torch.matmul(self.W_q, x)
        K = torch.matmul(self.W_k, x)

        output = x @ Q.transpose(-2, -1) @ K / N
        return output
    

class MergedSimpLinearAttn(nn.Module):
    def __init__(self, dim_embedding):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, 1/np.sqrt(dim_embedding), size=(dim_embedding, dim_embedding)))

    def forward(self, x):
        N = x.shape[-1]
        C = x @ x.transpose(-2, -1)

        output = C @ self.W @ x / N
        return output


def train_single(model, optimizer, criterion, train_input, train_label, test_input, test_label, epochs):
    train_losses = []
    test_losses = []

    for _ in tqdm(range(epochs)):
        output = model(train_input)
        prediction = output[:, -1, -1]
        loss = criterion(prediction, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        with torch.no_grad():
            output = model(test_input) 
            prediction = output[:, -1, -1]
            loss = criterion(prediction, test_label) 
            test_losses.append(loss.item())

    return train_losses, test_losses


def train(model, optimizer, criterion, train_input, train_label, test_input, test_label, epochs):

    for ep in range(epochs):
        output = model(train_input)
        prediction = output[:, -1, -1]
        loss = criterion(prediction, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep == epochs - 1:
            with torch.no_grad():
                output = model(test_input) 
                prediction = output[:, -1, -1]
                test_loss = criterion(prediction, test_label) 

    return loss.item(), test_loss.item()
