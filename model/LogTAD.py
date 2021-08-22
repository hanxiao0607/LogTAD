from . import DomainAdversarial
import torch.nn as nn
import torch.optim as optim
import torch
from utils.utils import get_center, epoch_time, get_dist
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
from gensim.models import Word2Vec

class LogTAD(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.emb_dim = options["emb_dim"]
        self.hid_dim = options["hid_dim"]
        self.output_dim = options["out_dim"]
        self.n_layers = options["n_layers"]
        self.dropout = options["dropout"]
        self.bias = options["bias"]
        self.weight_decay = options["weight_decay"]
        self.encoder = DomainAdversarial.DA_LSTM(self.emb_dim, self.hid_dim, self.output_dim, self.n_layers, self.dropout, self.bias)
        self.optimizer = optim.Adam(self.encoder.parameters(), weight_decay=self.weight_decay)
        self.device = options["device"]
        self.alpha = options["alpha"]
        self.max_epoch = options["max_epoch"]
        self.eps = options["eps"]
        self.source_dataset_name = options["source_dataset_name"]
        self.target_dataset_name = options["target_dataset_name"]
        self.loss_mse = nn.MSELoss()
        self.loss_cel = nn.CrossEntropyLoss()
        self.w2v = None
        self.center = None

    def _train(self, iterator, center):

        self.encoder.train()

        epoch_loss = 0

        for (i, batch) in enumerate(iterator):
            src = batch[0].to(self.device)
            domain_label = batch[1].to(self.device)
            labels = batch[2]
            self.optimizer.zero_grad()
            output, y_d = self.encoder(src, self.alpha)

            domain_label = domain_label.view(-1)
            center = center.to(self.device)

            mse = 0
            for (ind, val) in enumerate(output):
                if labels[ind] == 1:
                    mse += (10 - self.loss_mse(val, center))
                else:
                    mse += self.loss_mse(val, center)
            cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))
            loss = mse * 10e4 + cel
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

            center.cpu()
            src.cpu()
            domain_label.cpu()
            output.cpu()
            y_d.cpu()

        return epoch_loss / len(iterator)

    def _evaluate(self, iterator, center, epoch):

        self.encoder.eval()

        epoch_loss = 0

        lst_dist = []

        lst_mse = []
        lst_cel = []

        with torch.no_grad():
            for (i, batch) in enumerate(iterator):
                src = batch[0].to(self.device)
                domain_label = batch[1].to(self.device)
                labels = batch[2]
                output, y_d = self.encoder(src, self.alpha)
                if i == 0:
                    lst_emb = output
                else:
                    lst_emb = torch.cat((lst_emb, output), dim=0)

                domain_label = domain_label.view(-1)

                center = center.to(self.device)

                mse = 0
                for (ind, val) in enumerate(output):
                    if labels[ind] == 1:
                        mse += (10 - self.loss_mse(val, center))
                    else:
                        mse += self.loss_mse(val, center)

                cel = self.loss_cel(y_d, domain_label.to(dtype=torch.long))

                lst_mse.append(mse.detach().cpu().numpy())
                lst_cel.append(cel.detach().cpu().numpy())

                loss = mse * 10e4 + cel

                epoch_loss += loss.item()

                lst_dist.extend(get_dist(output, center))

                src.cpu()
                domain_label.cpu()
                lst_emb.cpu()
                output.cpu()
                y_d.cpu()

        if epoch < 10:
            center = get_center(lst_emb)
            print('get center:', center)
            center[(abs(center) < self.eps) & (center < 0)] = -self.eps
            center[(abs(center) < self.eps) & (center > 0)] = self.eps
            print('new center', center)

        print('\nmse:', np.mean(np.array(lst_mse)))
        print('cel:', np.mean(np.array(lst_cel)))
        return epoch_loss / len(iterator), center, lst_dist

    def train_LogTAD(self, train_iter, eval_iter, w2v):

        best_eval_loss = float('inf')

        for epoch in tqdm(range(self.max_epoch)):

            if epoch == 0:
                center = torch.Tensor([0.0 for _ in range(self.hid_dim)])
            if epoch > 9:
                center = fixed_center
            start_time = time.time()
            train_loss = self._train(self, train_iter, center)

            eval_loss, center, _ = self._evaluate(self, eval_iter, center, epoch)

            if epoch == 9:
                fixed_center = center

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if eval_loss < best_eval_loss and epoch >= 9:
                best_eval_loss = eval_loss
                torch.save(self.encoder.state_dict(), f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}.pt')

                self.center = fixed_center.cpu()
                pd.DataFrame(fixed_center.cpu().numpy()).to_csv(f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}_center.csv')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.10f}')
            print(f'\t Val. Loss: {eval_loss:.10f}')
        self.w2v = w2v
        w2v.save(f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')

    def load_model(self):
        self.w2v = Word2Vec.load(f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}_w2v.bin')
        self.encoder.load_state_dict(torch.load(f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}.pt'))
        self.center = torch.Tensor(
            pd.read_csv(f'/saved_model/{self.source_dataset_name}-{self.target_dataset_name}_center.csv', index_col=0).iloc[:, 0])
