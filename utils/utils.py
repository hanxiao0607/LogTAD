import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class LogDataset(Dataset):
    def __init__(self, features, domain_labels, labels):
        super().__init__()
        self.features = features
        self.domain_labels = domain_labels
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.features[idx], self.domain_labels[idx], self.labels[idx])

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def get_dist(ts, center):
    ts = ts.cpu().detach().numpy()
    center = center.cpu().numpy()
    temp = []
    for i in ts:
        temp.append(np.linalg.norm(i-center))
    return temp

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_center(emb, label = None):
    if label == None:
        return torch.mean(emb, 0)
    else:
        return 'Not defined'

def get_iter(X, y_d, y, batch_size = 1024, shuffle = True):
    dataset = LogDataset(X,y_d, y)
    if shuffle:
        iter = DataLoader(dataset, batch_size, shuffle=True, worker_init_fn=np.random.seed(42))
    else:
        iter = DataLoader(dataset, batch_size)
    return iter

def get_train_eval_iter(train_normal_s, train_normal_t, window_size=20, emb_dim=300):
    X = list(train_normal_s.Embedding.values)
    X.extend(list(train_normal_t.Embedding.values))
    X_new = []
    for i in tqdm(X):
        temp = []
        for j in i:
            temp.extend(j)
        X_new.append(np.array(temp).reshape(window_size, emb_dim))
    y_d = list(train_normal_s.target.values)
    y_d.extend(list(train_normal_t.target.values))
    y = list(train_normal_s.Label.values)
    y.extend(list(train_normal_t.Label.values))
    X_train, X_eval, y_d_train, y_d_eval, y_train, y_eval = train_test_split(X_new, y_d, y, test_size=0.2,
                                                                             random_state=42)
    X_train = torch.tensor(X_train, requires_grad=False)
    X_eval = torch.tensor(X_eval, requires_grad=False)
    y_d_train = torch.tensor(y_d_train).reshape(-1, 1).long()
    y_d_eval = torch.tensor(y_d_eval).reshape(-1, 1).long()
    y_train = torch.tensor(y_train).reshape(-1, 1).long()
    y_eval = torch.tensor(y_eval).reshape(-1, 1).long()
    train_iter = get_iter(X_train, y_d_train, y_train)
    eval_iter = get_iter(X_eval, y_d_eval, y_eval)
    return train_iter, eval_iter

def dist2label(lst_dist, R):
    y = []
    for i in lst_dist:
        if i <= R:
            y.append(0)
        else:
            y.append(1)
    return y
