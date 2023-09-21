import torch
import logging
import pandas as pd 
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import Classifier, SMILESModel, FASTAModel
from process_data import DTAData, CHARISOSMILEN, CHARPROTLEN, MACCSLEN
from train_and_test import train


MODEL_NAME = "FEDTA"
BATCH_SIZE = 256
DATASET = "davis"                                  

logging.basicConfig(filename=f'{MODEL_NAME}.log', level=logging.DEBUG)
writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    if DATASET == 'davis':
        df_train = pd.read_csv('data/Davis/Davis_train.csv')
        df_test = pd.read_csv('data/Davis/Davis_test.csv')
        max_smiles_len = 85
        max_fasta_len = 1000
    if DATASET == 'kiba':
        df_train = pd.read_csv('data/KIBA/KIBA_train.csv')
        df_test = pd.read_csv('data/KIBA/KIBA_test.csv')
        max_smiles_len = 100
        max_fasta_len = 1000
    if DATASET == 'Bind':
        df_train = pd.read_csv('data/BindingDB/BindingDB_train.csv')
        df_test = pd.read_csv('data/BindingDB/BindingDB_test.csv')
        max_smiles_len = 100
        max_fasta_len = 1000

    fasta_train = list(df_train['target_sequence'])
    fasta_test = list(df_test['target_sequence'])
    smiles_train = list(df_train['iso_smiles'])
    smiles_test = list(df_test['iso_smiles'])
    label_train = list(df_train['affinity'])
    label_test = list(df_test['affinity'])
    
    train_valid_set = DTAData(smiles_train, fasta_train, label_train, device, max_smiles_len, max_fasta_len)
    test_set = DTAData(smiles_test, fasta_test, label_test, device, max_smiles_len, max_fasta_len)

    train_size = int(len(train_valid_set) * 0.8)
    valid_size = len(train_valid_set) - train_size
    train_set, valid_set = random_split(train_valid_set, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    smiles_model = SMILESModel(char_set_len = MACCSLEN)
    fasta_model = FASTAModel(char_set_len=CHARPROTLEN+1)
    model = Classifier(smiles_model, fasta_model)
    model = model.to(device)

    train(model, train_loader, valid_loader, test_loader, writer, MODEL_NAME)


if __name__ == "__main__":
    main()