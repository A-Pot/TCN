import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../../")
from TCN.adding_problem.model import TCN

# Removed argparse so it's a easier to step through in an interactive environment
batch_size = 2
cuda = True # Assumes GPU is available
dropout = 0.0
clip = 1
epochs = 10
ksize = 2
levels = 8
seq_len = 400
log_interval = 100
lr = 4e-3
optimizer_type = 'Adam'
nhid = 30
seed = 1111

torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

input_channels = 2
n_classes = 3
LOW, HIGH = -20, 20 # Bounds for uniform ints

def random_int_generator():
    yield np.random.randint(LOW, HIGH)

# TODO: Generate datasets


# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [nhid]*levels
kernel_size = ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)

if cuda:
    model.cuda()
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()

optimizer = getattr(optim, optimizer_type)(model.parameters(), lr=lr)


def train(epoch):
    global lr
    model.train()
    batch_idx = 1
    total_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        if i + batch_size > X_train.size(0):
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            processed = min(i+batch_size, X_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size(0), 100.*processed/X_train.size(0), lr, cur_loss))
            total_loss = 0


def evaluate():
    model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item()


for ep in range(1, epochs+1):
    train(ep)
    tloss = evaluate()



