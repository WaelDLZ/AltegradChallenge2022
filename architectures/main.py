import sys 
(sys.path).append('D:\\OneDrive\\OneDrive - enpc.fr\\Documents\\Roman\\MVA\\ChallengeAltegrad\\AltegradChallenge2022')

from data import load_data, split_train_test
from datasets import DGLGraphDataset
from dgl.dataloading import GraphDataLoader
import torch
from networks import HGPSLModel
from train import train, test



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
    parser.add_argument('--patience', type=int, default=5, metavar='D',
                        help="number of epochs to wait to early stop the training")
    parser.add_argument('--print_every', type=int, default=1, metavar='D',
                        help="val_loss to print every X epochs")
    args = parser.parse_args()
    return args
                                                    


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print("Load data...")
    adj, features, edge_features = load_data(args.path_data)
    print('Data Loaded !')

    adj_train, features_train, edge_features_train, y_train, adj_test, features_test, edge_features_test, proteins_test = split_train_test(adj, features, edge_features, path=args.path_data)

    dataset_train = DGLGraphDataset(adj_train, features_train, edge_features_train, y_train)
    dataset_test = DGLGraphDataset(adj_test, features_test, edge_features_test, train=False)


    num_train = int(args.split_percent * len(dataset_train))
    num_val = len(dataset_train) - num_train

    train_set, val_set = torch.utils.data.random_split(dataset_train, [num_train, num_val])

    n_feat = features_train[0].shape[1]
    n_classes = args.n_classes
    n_hid = args.n_hid

    model = HGPSLModel(n_feat, n_classes, n_hid).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    print("Training...")
    bad_cound = 0
    best_val_loss = float("inf")
    best_epoch = 0
    train_times = []
    for e in range(args.epochs):
        train_loss = train(model, optimizer, train_loader, device)
        val_acc, val_loss = test(model, val_loader, device)
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
                "Epoch {}: val_loss={:.4f}, val_acc={:.4f}"
            )
            print(log_format.format(e + 1, val_loss, val_acc))
    print("Training done !")

if __name__ == "__main__":
    args = parse_args()
    main(args)





