import torch.nn.functional as F
import torch
from tqdm import tqdm

def train(model: torch.nn.Module, optimizer, trainloader, device, scheduler=None, weights=None):
    '''Function that runs a single epoch'''
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        _, out = model(batch_graphs, batch_graphs.ndata["feat"])
        if weights is not None:
            loss = F.nll_loss(out, batch_labels, weight=weights)
        else:
            loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if scheduler:
        scheduler.step()
    return total_loss / num_batches

def train_multimodal(model: torch.nn.Module, optimizer, trainloader, device, scheduler=None):
    '''Function that runs a single epoch for the multimodal setting'''
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        batch_graphs, protein_embeddings, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        protein_embeddings = protein_embeddings.to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"], protein_embeddings)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if scheduler:
        scheduler.step()

    return total_loss / num_batches

def train_multi_graph(model: torch.nn.Module, optimizer, trainloader, device, scheduler=None):
    '''Function that runs a single epoch for the multigraphs settings'''
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in tqdm(trainloader):
        optimizer.zero_grad()
        list_g, batch_labels = batch
        list_g = [g.to(device) for g in list_g]
        batch_labels = batch_labels.long().to(device)
        _, out = model(list_g)
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if scheduler:
        scheduler.step()
    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    '''Function that computes evaluation metrics'''
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        _, out = model(batch_graphs, batch_graphs.ndata["feat"])
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs

def test_multimodal(model: torch.nn.Module, loader, device):
    '''Function that computes evaluation metrics for the multimodal setting'''
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, protein_embeddings, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        protein_embeddings = protein_embeddings.to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"], protein_embeddings)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs

@torch.no_grad()
def test_multi_graph(model: torch.nn.Module, loader, device):
    '''Function that computes evaluation metrics for the multi graph setting'''
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        list_g, batch_labels = batch
        list_g = [g.to(device) for g in list_g]
        num_graphs += batch_labels.size(0)
        batch_labels = batch_labels.long().to(device)
        _, out = model(list_g)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
    return correct / num_graphs, loss / num_graphs
