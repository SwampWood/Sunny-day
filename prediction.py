import torch
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchsummary import summary
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler
from format import column_sql, headers


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
period = 120
models = {
    'Температура': 'models/temp'
}

class LRScheduler:
    def __init__(self, optimizer, patience=3, min_lr=1e-6, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True)

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class MiniDataset(Dataset):
    def __init__(self, database, len_batches):
        self.scaler = MinMaxScaler()
        database = torch.flatten(torch.from_numpy(
            self.scaler.fit_transform(database[database != 'NULL'].astype(np.float32).reshape(-1, 1)))).to(device)
        print(database)
        self.len_batch = len_batches
        self.database = database
        self.targets = database[len_batches:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.database[idx:idx + self.len_batch]
        y = self.targets[idx][None]
        return x, y

    def reverse(self, x):
        return self.scaler.inverse_transform(x)
    
    def to_form(self, x):
        return torch.flatten(torch.from_numpy(
            self.scaler.transform(x.astype(np.float32).reshape(-1, 1)))).to(device)


class LargeDataset(Dataset):
    def __init__(self, database, len_batches=period, params=tuple(range(20))):
        database = np.array(database)
        self.len_batches = len_batches
        self.data = {params[i]: database[:, i] for i in range(len(params))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return MiniDataset(self.data[idx], self.len_batches)


def train(model, dataloader, loss_function, optimizer=None):
    model.train()
    total_loss = 0
    for images, labels in tqdm(dataloader):
        outputs = model(images)
        loss = loss_function(outputs, labels)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def test_(model, dataloader, loss_function):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            outputs = model(images)

            loss = loss_function(outputs, labels)

            total_loss += loss.item()

            y_pred.extend(outputs.tolist())
            y_true.extend(labels.tolist())

    return total_loss / len(dataloader)


def show_losses(train_loss_hist, test_loss_hist):
    clear_output()
    objects = np.array([train_loss_hist, test_loss_hist]).T

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.legend(plt.plot(list(range(len(train_loss_hist))), objects),
               ('train', 'test'))
    plt.yscale('log')
    plt.grid()

    plt.show()


def evaluate(model, test_loader, loss_function):
    y_true, y_pred, _ = test_(model, test_loader, loss_function)

    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1


def run_train_loop(model, optimiser, loss_func, train_loader, val_loader,
                   num_epochs, LRscheduler=None, early_stop=None,
                   save_model=None):
    train_hist = []
    val_hist = []
    for e in range(num_epochs):
        print("Training...")
        train_loss = train(model, train_loader, loss_func, optimiser)
        if save_model:
            torch.save(model.state_dict(), save_model)
        train_hist.append(train_loss)
        print("Validating...")
        val_loss = test_(model, val_loader, loss_func)
        val_hist.append(val_loss)
        print(train_hist, val_hist)
        clear_output()
        show_losses(train_hist, val_hist)
        if LRscheduler:
            LRscheduler(train_loss)
        if early_stop:
            early_stop(train_loss)
            if early_stop.early_stop:
                print('INFO: Early stopping. Epoch:', e + 1)
                break


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x[None]
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)

        return out


def create_prediction(type, data, le):
    model = LSTMModel(period, 64, 3, 1, 0.2).to(device)
    model.load_state_dict(torch.load(models[type]))
    model.eval()
    temp = data[len(data) - period:]
    dt = MiniDataset(data, period)
    res = []
    for i in range(le):
        res.append(dt.reverse(dt.to_form(temp)))
        temp = temp[1:] + [res[-1]]
    return res


if __name__ == '__main__':
    whole_data = LargeDataset(column_sql('drive/MyDrive/temporary.db'), params=tuple(headers.keys()))
    temp_data = whole_data['ТЕМВОЗДМ']
    train_dataset, test_dataset = random_split(temp_data, [0.9, 0.1])
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    model = LSTMModel(period, 64, 3, 1, 0.2).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss(reduction='mean')

    train_dl = DataLoader(train_dataset, BATCH_SIZE, True)
    test_dl = DataLoader(test_dataset, BATCH_SIZE, False)

    lr_scheduler = LRScheduler(optim, min_lr=1e-9)

    run_train_loop(
        model,
        optim,
        loss,
        train_dl,
        test_dl,
        NUM_EPOCHS,
        lr_scheduler,
        save_model='model',
    )