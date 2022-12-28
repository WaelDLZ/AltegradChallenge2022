import torch.nn.functional as F
import torch
from tqdm import tqdm

def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs