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
from format import column_sql, headers


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
        database = database[database != 'NULL'].astype(np.float64)
        self.inputs = database[:len_batches]
        for i in tqdm(range(1, len(database) - len_batches - 1)):
            self.inputs = np.vstack((self.inputs, database[i:i + len_batches]))
        self.targets = database[len_batches:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y


class LargeDataset(Dataset):
    def __init__(self, database, len_batches=120, params=tuple(range(20))):
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

            pred = outputs.argmax(dim=2, keepdim=True)
            y_pred.extend(pred.view_as(labels).tolist())
            y_true.extend(labels.tolist())

    return y_true, y_pred, total_loss / len(dataloader)


def show_losses(train_loss_hist, test_loss_hist):
    clear_output()
    objects = np.array([train_loss_hist, test_loss_hist]).T

    plt.figure(figsize=(12,4))

    plt.subplot(1, 2, 1)
    plt.legend(plt.plot(np.arange(len(train_loss_hist)), objects),
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
    train_hist = np.array([])
    val_hist = np.array([])
    for e in range(num_epochs):
        print("Training...")
        train_loss = train(model, train_loader, loss_func, optimiser)
        if save_model:
            torch.save(model.state_dict(), save_model)
        train_hist = np.append(train_hist, np.array([train_loss]))
        print("Validating...")
        val_loss = test_(model, val_loader, loss_func)
        val_hist = np.append(val_hist, np.array([val_loss]))
        clear_output()
        show_losses(train_hist, val_hist)
        if LRscheduler:
            LRscheduler(train_loss)
        if early_stop:
            early_stop(train_loss)
            if early_stop.early_stop:
                print('INFO: Early stopping. Epoch:', e + 1)
                break


class LinearModel(nn.Module):
    def __init__(self, len_batch):
        super().__init__()
        self.fc1 = nn.Linear(len_batch, 256)
        self.fc2 = nn.Linear(256, 64)
        self.drop_out = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = self.drop_out(out)
        out = nn.functional.relu(self.fc3(out))
        return out


if __name__ == '__main__':
    whole_data = LargeDataset(column_sql('database/temporary.db'), params=tuple(headers.keys()))
    temp_data = whole_data['ТЕМВОЗДМ']
    train_dataset, test_dataset = random_split(temp_data, [0.9, 0.1])
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    model = LinearModel(120)

    optim = torch.optim.Adam(model.parameters(), lr=1e-6)
    loss = nn.MSELoss()

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
        save_model='lungs_model10',
    )