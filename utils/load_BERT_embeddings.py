import pickle

def load_BERT_embedding(path):
    embeddings = pickle.load(open(path, 'rb'))
    for i in range(len(embeddings['Amino Acids'])):
        embeddings['Amino Acids'][i] = embeddings['Amino Acids'][i][1:-1]
    return embeddings['Amino Acids'], embeddings['Protein']
