import sys 
#(sys.path).append('D:\\OneDrive\\OneDrive - enpc.fr\\Documents\\Roman\\MVA\\ChallengeAltegrad\\AltegradChallenge2022')
(sys.path).append('C:\\Users\\Wael\\Desktop\\MVA\\Altegrad\\altegrad_challenge_2022\\AltegradChallenge2022')

from load_BERT_embeddings import load_BERT_embedding
import pickle
from data import split_train_test, load_sequences
from datasets import DGLGraphDataset_Multimodal
from dgl.dataloading import GraphDataLoader
import torch
from networks import HGPSLModel, MultimodalModel
from train import train_multimodal, test_multimodal
import numpy as np
import csv
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="HGP-SL-DGL altegrad")

    parser.add_argument('--path_data', type=str, default='', metavar='D',
                        help="folder where data is located")
    parser.add_argument('--split_percent', type=float, default=0.8, metavar='D',
                        help="percentage of train data of the split train/val, default is 80%")
    parser.add_argument('--n_classes', type=int, default=18, metavar='D',
                        help="number of classes in classification task")
    parser.add_argument('--in_feat_multimodal', type=int, default=256, metavar='D',
                        help="dimension of input of multimodal network")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='D',
                        help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, metavar='D',
                        help="weight_decay")
    parser.add_argument('--batch_size', type=int, default=64, metavar='D',
                        help="batch_size")
    parser.add_argument('--epochs', type=int, default=1, metavar='D',
                        help="epochs")
    parser.add_argument('--path_save_model', type=str, default='', metavar='D',
                        help="where to save the model")
    parser.add_argument('--patience', type=int, default=5, metavar='D',
                        help="number of epochs to wait to early stop the training")
    parser.add_argument('--print_every', type=int, default=1, metavar='D',
                        help="val_loss to print every X epochs")
    parser.add_argument('--path_embeddings', type=str,
                        help="path to the BERT embeddings pickle file")
    parser.add_argument('--path_submission', type=str,
                        help="path to csv file for submissions")
    parser.add_argument('--path_graph_model', type=str, help='path to graph model')
    parser.add_argument('--n_hid_graph_model', type=int, default=128)
    parser.add_argument('--path_pretrained_model', type=str,
                        help="Path to pretrained models weights")
    parser.add_argument('--training', type=str, default='last_layers', 
                        help="defines training strategy can be all(graph and last layers are trained of just last_layers")
    args = parser.parse_args()
    return args
                                                    


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(29)

    print("Load data...")
    adj = pickle.load(open(args.path_data+'adj.pkl', 'rb'))
    features = pickle.load(open(args.path_data + 'features.pkl', 'rb'))
    edge_features = pickle.load(open(args.path_data + 'edge_features.pkl', 'rb'))
    print('Data Loaded !')

    adj_train, features_train, edge_features_train, y_train, adj_test, features_test,\
        edge_features_test, proteins_test = split_train_test(adj, features, edge_features, path=args.path_data)
    
    features_train, _ = load_BERT_embedding(args.path_embeddings + '/train/embeddings.pkl')
    features_test, _ = load_BERT_embedding(args.path_embeddings + '/test/embeddings.pkl')

    sequences_train, sequences_test, proteins_test, y_train = load_sequences(args.path_data)
    vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
    protein_train_embeddings = vec.fit_transform(sequences_train)
    protein_test_embeddings = vec.transform(sequences_test)
    
    dataset_train = DGLGraphDataset_Multimodal(adj_train, features_train, edge_features_train,
                                               protein_train_embeddings, y_train)
    dataset_test = DGLGraphDataset_Multimodal(adj_test, features_test, edge_features_test,
                                              protein_test_embeddings, train=False)

    num_train = int(args.split_percent * len(dataset_train))
    num_val = len(dataset_train) - num_train

    train_set, val_set = torch.utils.data.random_split(dataset_train, [num_train, num_val],
                                                       generator=torch.Generator().manual_seed(42))

    n_feat = features_train[0].shape[1]
    dim_protein_embedding = protein_train_embeddings[0].shape[0]

    graph_model = HGPSLModel(n_feat, args.n_classes, args.n_hid_graph_model).to(device)
    model_multi = MultimodalModel(graph_model=graph_model, n_classes=args.n_classes,
                                  h_dim=args.in_feat_multimodal,
                                  dim_protein_embedding=dim_protein_embedding,
                                  dim_graph_embedding=args.n_hid_graph_model).to(device)

    if args.path_graph_model:
        graph_model.load_state_dict(torch.load(args.path_graph_model))
    if args.path_pretrained_model :
        model_multi.load_state_dict(torch.load(args.path_pretrained_model))

    if args.training == 'last_layers':
        for param in model_multi.graph_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model_multi.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = GraphDataLoader(dataset_test, batch_size=1, shuffle=False)

    if args.epochs > 0:
        print("Training...")
        bad_cound = 0
        best_val_loss = float("inf")
        best_epoch = 0
        train_times = []
        for e in range(args.epochs):
            train_loss = train_multimodal(model_multi, optimizer, train_loader, device, scheduler=scheduler)
            val_acc, val_loss = test_multimodal(model_multi, val_loader, device)
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                bad_cound = 0
                torch.save(model_multi.state_dict(), args.path_save_model)
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

        model_multi.eval()
        preds = list()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                x, prot_emb = batch
                x = x.to(device)
                prot_emb = prot_emb.to(device)
                out = model_multi(x, x.ndata["feat"], prot_emb)
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



args = parse_args()
main(args)





