import sys

#(sys.path).append('D:\\OneDrive\\OneDrive - enpc.fr\\Documents\\Roman\\MVA\\ChallengeAltegrad\\AltegradChallenge2022')
(sys.path).append('C:\\Users\\Wael\\Desktop\\MVA\\Altegrad\\altegrad_challenge_2022\\AltegradChallenge2022')

from utils.load_BERT_embeddings import load_BERT_embedding
from data import load_data, split_train_test
from utils.datasets import DGLGraphDataset_ngraphs
from dgl.dataloading import GraphDataLoader
import torch
from architectures.networks import GNN_multiple_roman, GNN_multiple
from utils.train import train_multi_graph, test_multi_graph
import numpy as np
import csv
from tqdm import tqdm
import pickle
from utils.utils import classes_weights

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HGP-SL-DGL altegrad")

    parser.add_argument('--path_data', type=str, default='', metavar='D',
                        help="folder where data is located")
    parser.add_argument('--split_percent', type=float, default=0.8, metavar='D',
                        help="percentage of train data of the split train/val, default is 80%")
    parser.add_argument('--n_classes', type=int, default=18, metavar='D',
                        help="number of classes in classification task")
    parser.add_argument('--n_hid', type=int, default=128, metavar='D',
                        help="dimension of hidden layer in graph network")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='D',
                        help="weight_decay")
    parser.add_argument('--batch_size', type=int, default=64, metavar='D',
                        help="batch_size")
    parser.add_argument('--epochs', type=int, default=50, metavar='D',
                        help="epochs")
    parser.add_argument('--path_save_model', type=str, default='', metavar='D',
                        help="where to save the model")
    parser.add_argument('--patience', type=int, default=10, metavar='D',
                        help="number of epochs to wait to early stop the training")
    parser.add_argument('--print_every', type=int, default=1, metavar='D',
                        help="val_loss to print every X epochs")
    parser.add_argument('--path_embeddings', type=str,
                        help="path to the BERT embeddings pickle file")
    parser.add_argument('--path_submission', type=str,
                        help="path to csv file for submissions")
    parser.add_argument('--path_pretrained_model', type=str,
                        help="Path to pretrained models weights")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Dropout ratio")
    parser.add_argument('--graph_layers', type=str, default=None,
                        help="Type of message passing layers in GNN")
    parser.add_argument('--whole_graph', type=bool, default=False,
                        help="")
    parser.add_argument('--filter_edges', type=list, default=[True, True, False, False],
                        help="")                   
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(3407)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    print("Load data...")
    adj = pickle.load(open(args.path_data + 'adj.pkl', 'rb'))
    features = pickle.load(open(args.path_data + 'features.pkl', 'rb'))
    edge_features = pickle.load(open(args.path_data + 'edge_features.pkl', 'rb'))

    #adj, features, edge_features = load_data(path=args.path_data)
    print('Data Loaded !')

    adj_train, features_train, edge_features_train, y_train, adj_test, features_test, \
    edge_features_test, proteins_test = split_train_test(adj, features, edge_features, path=args.path_data)

    features_train, _ = load_BERT_embedding(args.path_embeddings + '/train/embeddings.pkl')
    features_test, _ = load_BERT_embedding(args.path_embeddings + '/test/embeddings.pkl')

    dataset_train = DGLGraphDataset_ngraphs(adj_train, features_train, edge_features_train, y_train,
                                            whole_graph=args.whole_graph, filter_edges=args.filter_edges)
    dataset_test = DGLGraphDataset_ngraphs(adj_test, features_test, edge_features_test, train=False,
                                           whole_graph=args.whole_graph, filter_edges=args.filter_edges)
    num_train = int(args.split_percent * len(dataset_train))
    num_val = len(dataset_train) - num_train

    train_set, val_set = torch.utils.data.random_split(dataset_train, [num_train, num_val],
                                                       generator=torch.Generator().manual_seed(42))

    in_feat = features_train[0].shape[1]
    n_classes = args.n_classes
    n_hid = args.n_hid
    num_graphs = args.whole_graph + sum(args.filter_edges)

    # model = GNN_multiple(in_feat, n_classes, n_hid, dropout=args.dropout, graph_layers=args.graph_layers).to(device)

    model = GNN_multiple_roman(in_feat, n_classes, n_hid, dropout=args.dropout, graph_layers=args.graph_layers, num_graphs=num_graphs).to(device)


    if args.path_pretrained_model:
        model.load_state_dict(torch.load(args.path_pretrained_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(dataset_test, batch_size=1, shuffle=False)

    if args.epochs > 0:
        print("Training...")
        bad_cound = 0
        best_val_loss = float("inf")
        best_epoch = 0
        train_times = []
        for e in range(args.epochs):
            train_loss = train_multi_graph(model, optimizer, train_loader, device, scheduler=scheduler)
            val_acc, val_loss = test_multi_graph(model, val_loader, device)
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                bad_cound = 0
                torch.save(model.state_dict(), args.path_save_model)
            else:
                bad_cound += 1
            if bad_cound >= args.patience:
                break

            if (e + 1) % args.print_every == 0:
                log_format = (
                    "Epoch {}:  train_loss={:.4f}, val_loss={:.4f}, val_acc={:.4f}"
                )
                print(log_format.format(e + 1, train_loss, val_loss, val_acc))
        print("Training done !")

    if args.path_submission:
        print("Inference on test set: ")

        model.eval()
        preds = list()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                list_g = batch
                list_g = [g.to(device) for g in list_g]
                _, out = model(list_g)
                pred = out.detach().cpu().numpy()
                pred = np.exp(pred)
                preds.append(pred)
        y_pred_proba = np.array([list(pred[0]) for pred in preds])

        print("Writing submissions to : {}".format(args.path_submission))
        with open(args.path_submission, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            lst = list()
            for i in range(18):
                lst.append('class' + str(i))
            lst.insert(0, "name")
            writer.writerow(lst)
            for i, protein in enumerate(proteins_test):
                lst = y_pred_proba[i, :].tolist()
                lst.insert(0, protein)
                writer.writerow(lst)


if __name__ == "__main__":
    args = parse_args()
    main(args)





