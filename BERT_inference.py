"""
Author: Wael DOULAZMI // 26.12.2022
Purpose: Use prot_bert to compute embeddings of all our proteins that can be used later
"""
from transformers import BertModel, BertTokenizer
import re
from data import load_sequences
from tqdm import tqdm
import torch
import os
import json

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

# Load sequences

sequences_train, sequences_test = load_sequences()

# Put spaces between amino acids

sequences_train = [" ".join(seq) for seq in sequences_train]
sequences_test = [" ".join(seq) for seq in sequences_test]

# Replace U Z O B with X

sequences_train = [re.sub(r"UZOB", 'X', seq) for seq in sequences_train]
sequences_test = [re.sub(r"UZOB", 'X', seq) for seq in sequences_test]


# Encode
train_bert_embeddings = {"Protein": [], "Amino Acids": []}
test_bert_embeddings = {"Protein": [], "Amino Acids": []}

model.eval()
with torch.no_grad():

    # Get Embeddings per amino acid and per protein
    for seq in tqdm(sequences_train):
        encoded_input = tokenizer(seq, return_tensors='pt')
        output = model(**encoded_input)
        prot_embedding = output.pooler_output.detach().numpy()[0]
        amino_embeddings = output.last_hidden_state.detach().numpy()[0]
        
        train_bert_embeddings["Protein"].append(prot_embedding)
        train_bert_embeddings["Amino Acids"].append(amino_embeddings)

    for seq in tqdm(sequences_test):
        encoded_input = tokenizer(seq, return_tensors='pt')
        output = model(**encoded_input)
        prot_embedding = output.pooler_output.detach().numpy()[0]
        amino_embeddings = output.last_hidden_state.detach().numpy()[0]
    
        test_bert_embeddings["Protein"].append(prot_embedding)
        test_bert_embeddings["Amino Acids"].append(amino_embeddings)


# Save results once for all

os.makedirs('data/bert_embeddings/train/', exist_ok=True)
os.makedirs('data/bert_embeddings/test/', exist_ok=True)

with open('data/bert_embeddings/train/embeddings.json', 'w') as fp:
    json.dump(train_bert_embeddings, fp)

with open('data/bert_embeddings/test/embeddings.json', 'w') as fp:
    json.dump(test_bert_embeddings, fp)
