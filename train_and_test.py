import torch
import copy
import logging
from metrics import get_cindex, get_rm2
import os

def test(data_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pred = []
        score_list = []
        label_list = []
        smiles_list = []
        fasta_list = []

        running_loss = 0.0
        for sample in data_loader:
            smiles, fasta, label = sample
            score = model(smiles, fasta).view(-1)
            loss = loss_fn(score, label)
            running_loss += loss.item()
            y_pred += score.detach().cpu().tolist()
            y_true += label.detach().cpu().tolist()
            score_list.append(score)
            label_list.append(label)
            smiles_list.append(smiles)
            fasta_list.append(fasta)
        with open( "prediction.txt", 'a') as f:
            for i in range(len(score_list)):
                f.write(str(smiles_list[i]) + " " + str(fasta_list[i]) + " " + str(label_list[i]) + " " + str(score_list[i]) +'\n')
        ci = get_cindex(y_true, y_pred)
        rm2 = get_rm2(y_true, y_pred)
        model.train()
    return running_loss/len(data_loader), ci, rm2


def train(model, train_loader, val_loader, test_loader, writer, NAME, lr=0.0001, epoch=150):
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.MSELoss()
    model_best = copy.deepcopy(model)
    min_loss = 1000

    for epo in range(epoch):
        model.train()
        running_loss = 0.0
        
        val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn)
        for data in train_loader:
            smiles, fasta, label = data
            score = model(smiles, fasta).view(-1)
            loss = loss_fn(score, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
        writer.add_scalar(f'Loss/train_{NAME}', running_loss/len(train_loader), epo)
        logging.info(f'Training at Epoch {epo + 1} with loss {running_loss/len(train_loader):.4f}')
        
        val_loss, val_ci, val_rm2 = test(val_loader, model, loss_fn)
        writer.add_scalar(f'Loss/valid_{NAME}', val_loss, epo)
        logging.info(f'Validation at Epoch {epo+1} with loss {val_loss:.4f}, ci {val_ci}, rm2 {val_rm2}')
        if val_loss < min_loss:
            min_loss = val_loss
            model_best = copy.deepcopy(model)

    test_loss, test_ci, test_rm2 = test(test_loader, model_best, loss_fn)
    logging.info(f'Test loss {test_loss:.4f}, ci {test_ci}, rm2 {test_rm2}')
    torch.save(model_best, f'model/{NAME}.pth')

