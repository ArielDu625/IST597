import os
import torch
import torch.optim as optim
import torch.nn as nn
from time import time
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from util import GraphExprDataset
from model.gnn import Encoder, GraphClassifier

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda')

class Model:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _count_paras(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model(self):
        hid_dim = 256
        dropout = 0.5
        device = DEVICE

        model = GraphClassifier(self.input_dim, hid_dim, self.output_dim, device).to(device)
    
        print()
        print('=======================================')
        print(f'Input dim:  {self.input_dim}')
        print(f'output dim: {self.output_dim}')
        print(f'Hidden dim: {hid_dim}')
        print(f'Count trainable paras: {self._count_paras(model):,}')

        return model


class Trainer:
    def __init__(self, model, epochs):
        self.model = model
        self.epochs = epochs
        self.device = DEVICE
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-6)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.criterion = nn.BCEWithLogitsLoss()

    def _train(self, data_loader):
        self.model.train()
        epoch_loss = 0

        for i, data in enumerate(data_loader):
            data.to(self.device)
            # our = [batch_size, output_dim]
            out = self.model(data)
            out = out.reshape(-1)
            # target = [batch_size]
            target = data.y.to(self.device)
            target = target.float()
            
            loss = self.criterion(out, target)  # Compute the loss.
            self.optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Derive gradients.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  
            self.optimizer.step()  # Update parameters based on gradients.
            

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)
    
    def _evaluate(self, data_loader):
        self.model.eval()    
        epoch_loss = 0
        
        y_pred = []
        y_true = []
        with torch.no_grad():    
            for i, data in enumerate(data_loader):
                data.to(self.device)
                out = self.model(data) 
                out = out.reshape(-1)

                target = data.y.to(self.device)
                target = target.float()
                
                loss = self.criterion(out, target)            
                epoch_loss += loss.item()

                y_pred.append(torch.round(torch.sigmoid(out)))
                y_true.append(target)
        
        y_pred = torch.hstack(y_pred).cpu()
        y_true = torch.hstack(y_true).cpu()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, average='binary')
        rcl = recall_score(y_true, y_pred, pos_label=1, average='binary')

        # loss = epoch_loss / len(data_loader)

        return epoch_loss, acc, prec, rcl

    def training(self, train_iter, valid_iter, model_name):
        print()
        print('Training...')
        print('=======================================')
        best_valid_loss = float('inf')
        best_metrics = {}
        # patience for early stopping
        patience = 20
        trigger = 0
        total_train_time = 0

        for epoch in range(self.epochs):
            start_time = time()
            train_loss = self._train(train_iter)
            total_train_time += time() - start_time

            train_loss, train_acc, train_prec, train_rcl = self._evaluate(train_iter)
            valid_loss, val_acc, val_prec, val_rcl = self._evaluate(valid_iter)
            print(f'Epoch: {epoch+1}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, train prec: {train_prec:.4f}, train rcl: {train_rcl:.4f}, val loss: {valid_loss:.4f}, val acc: {val_acc:.4f}, val prec: {val_prec:.4f}, val rcl: {val_rcl:.4f}')

            if valid_loss < best_valid_loss:
                trigger = 0
                best_valid_loss = valid_loss
                best_metrics.update({
                    "acc": val_acc,
                    "prec": val_prec,
                    "rcl": val_rcl
                })
                torch.save(model.state_dict(), f'saved_models/{model_name}.pt')
            # else:
            #     trigger += 1
            #     if trigger >= patience:
            #         break
            self.scheduler.step()
        print(f'Best valid loss: {best_valid_loss:.4f}')
        print(f'Best metrics: {best_metrics}')
        print(f'Training time per epoch: {total_train_time/self.epochs:.4f}')


if __name__ == '__main__':
    """
    Accept 3 parameters:
        train: whether to train the model.
        batch_size: batch size.
        epochs: epochs
    """
    parser = ArgumentParser(description='GraphMR')
    parser.add_argument('--train', type=int, default=1, help='whether to train the model')
    parser.add_argument('--batch_size', type=int, default=128, help='size of each mini batch')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')

    args = parser.parse_args()

    # === 1. Get dataset ==============================
    # Before creating dataset, destination folder should be empty
    os.system(f'rm -rf dataset/processed/*')

    is_dag = False
    valid_depth = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    dataset = GraphExprDataset('dataset', '40k_train.json', '40k_test.json', '40k_val_shallow.json', is_dag, valid_depth)
    train_set = dataset[:dataset.train_size]
    test_set = dataset[dataset.train_size: (dataset.train_size+dataset.test_size)]
    valid_set = dataset[(-dataset.val_size):]
    
    print()
    print(f'===================================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of training graphs: {len(train_set)}')
    print(f'Number of testing graphs: {len(test_set)}')
    print(f'Number of valid graphs: {len(valid_set)}')
    print(f'Symbol vocab: {dataset.symbol_vocab}')
    print(f'DAG symbol vocab: {dataset.dag_symbol_vocab}')
    print(f'Batch size: {args.batch_size}')


    # # === 2. Get model ==============================
    input_dim = dataset.num_node_features
    output_dim = 1
    model = Model(input_dim, output_dim).build_model()

    if is_dag:
        model_name = 'graphmr_dag'
    else:
        model_name = 'graphmr_ast'
    # # === 3. Training ==============================
    if args.train:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

        trainer = Trainer(model, args.epochs)
        trainer.training(train_loader, valid_loader, model_name)

    # === 4. Verifing ==============================
    print()
    print('Verification...')
    print('=======================================')

    model.load_state_dict(torch.load(f'saved_models/{model_name}.pt'))
    model.eval()

    preds = []
    y_true = []
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data.to(DEVICE)
            out = model(data)
            out = out.reshape(-1)
            out = torch.round(torch.sigmoid(out))
            preds.append(out)
            y_true.append(data.y)

    y_true = torch.hstack(y_true).cpu()
    preds = torch.hstack(preds).cpu()
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, pos_label=1, average='binary')
    rcl = recall_score(y_true, preds, pos_label=1, average='binary')
    print(f"test acc = {acc:4f}, test prec: {prec:.4f}, test rcl: {rcl:.4f}")

