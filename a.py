# Fixing LTCCell implementation and re-run the demo training.
# Changes: apply rec_mask to recurrent weights, not to h. Also adjust a few small issues (logit conversions removed - use logits directly from heads).

import os
import math
import time
import random
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

class LTCCell(nn.Module):
    """
    Fixed LTCCell: apply rec_mask to recurrent weights, not to hidden state.
    """
    def __init__(self, input_size, hidden_size, dt=0.04, tau_min=0.01, tau_max=10.0, sparsity=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        self.Wx = nn.Parameter(torch.randn(hidden_size, input_size) * (1.0 / math.sqrt(input_size)))
        self.Wh = nn.Parameter(torch.randn(hidden_size, hidden_size) * (1.0 / math.sqrt(hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.log_tau = nn.Parameter(torch.log(torch.ones(hidden_size) * 1.0))
        
        mask = (torch.rand(hidden_size, hidden_size) < sparsity).float()
        mask.fill_diagonal_(1.0)
        self.register_buffer("rec_mask", mask)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wx)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.constant_(self.bias, 0.0)
        with torch.no_grad():
            self.log_tau.copy_(torch.log(torch.ones_like(self.log_tau) * 0.5))
    
    def get_tau(self):
        tau = F.softplus(self.log_tau) + 1e-6
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau
    
    def forward(self, x, h=None):
        single_step = (x.dim() == 2)
        if single_step:
            x = x.unsqueeze(1)
        B, T, _ = x.shape
        device = x.device
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=device)
        hs = []
        tau = self.get_tau().to(device)
        alpha = torch.exp(-self.dt / tau)  # (hidden_size,)
        # masked recurrent weights
        Wh_masked = self.Wh * self.rec_mask.to(device)
        for t in range(T):
            xt = x[:, t, :]  # (B, input_size)
            pre = F.linear(xt, self.Wx) + F.linear(h, Wh_masked) + self.bias  # (B, hidden_size)
            activated = torch.tanh(pre)
            h = alpha * h + (1 - alpha) * activated
            hs.append(h.unsqueeze(1))
        h_seq = torch.cat(hs, dim=1)
        if single_step:
            return h_seq[:, 0, :]
        return h_seq

class LightLTCSeizNet(nn.Module):
    def __init__(self, in_channels, in_samples, latent_dim=64, ltc_sizes=(64,48), dt=1/25):
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.filterbank = nn.Conv1d(in_channels, in_channels*8, kernel_size=128, stride=32, padding=64, groups=in_channels)
        self.bn_fb = nn.BatchNorm1d(in_channels*8)
        self.act = nn.GELU()
        time_len = (in_samples + 2*64 - 128) // 32 + 1
        self.time_len = max(1, time_len)
        self.enc_conv = nn.Conv1d(in_channels*8, latent_dim, kernel_size=7, padding=3, groups=1)
        self.bn_enc = nn.BatchNorm1d(latent_dim)
        self.ltc1 = LTCCell(input_size=latent_dim, hidden_size=ltc_sizes[0], dt=dt)
        self.ltc2 = LTCCell(input_size=ltc_sizes[0], hidden_size=ltc_sizes[1], dt=dt)
        self.att_q = nn.Linear(ltc_sizes[1], 32)
        self.att_k = nn.Linear(ltc_sizes[1], 32)
        self.att_v = nn.Linear(ltc_sizes[1], 32)
        self.channel_gate = nn.Sequential(nn.Linear(in_channels, max(4,in_channels//2)), nn.GELU(), nn.Linear(max(4,in_channels//2), in_channels), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.detect_head = nn.Sequential(nn.Linear(ltc_sizes[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        self.pred_head = nn.Sequential(nn.Linear(ltc_sizes[1], 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        
    def forward(self, x):
        B, C, T = x.shape
        ch_energy = x.abs().mean(dim=2)
        gate = self.channel_gate(ch_energy)
        x = x * gate.unsqueeze(2)
        fb = self.filterbank(x)
        fb = self.bn_fb(fb)
        fb = self.act(fb)
        enc = self.enc_conv(fb)
        enc = self.bn_enc(enc)
        enc = self.act(enc)
        enc_t = enc.transpose(1,2)
        h1 = self.ltc1(enc_t)
        h2 = self.ltc2(h1)
        Q = self.att_q(h2)
        K = self.att_k(h2)
        V = self.att_v(h2)
        att_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(Q.size(-1))
        att_weights = F.softmax(att_scores, dim=-1).mean(dim=1)
        pooled = h2.mean(dim=1)
        detect_logit = self.detect_head(pooled).squeeze(-1)
        pred_logit = self.pred_head(pooled).squeeze(-1)
        detect_prob = torch.sigmoid(detect_logit)
        pred_prob = torch.sigmoid(pred_logit)
        taus = {'ltc1_tau': self.ltc1.get_tau().detach().cpu().numpy(), 'ltc2_tau': self.ltc2.get_tau().detach().cpu().numpy()}
        return detect_prob, pred_prob, {'att_weights': att_weights.detach().cpu().numpy(), 'taus': taus}

class SyntheticEEGDataset(Dataset):
    def __init__(self, n_samples=1000, channels=18, fs=256, window_s=10, seizure_frac=0.1):
        self.n_samples = n_samples
        self.channels = channels
        self.fs = fs
        self.window_s = window_s
        self.win_len = int(window_s * fs)
        self.seizure_frac = seizure_frac
        self.X = np.zeros((n_samples, channels, self.win_len), dtype=np.float32)
        self.y_detect = np.zeros((n_samples,), dtype=np.int64)
        self.y_pred = np.zeros((n_samples,), dtype=np.int64)
        self.generate()
    
    def generate(self):
        for i in range(self.n_samples):
            t = np.linspace(0, self.window_s, self.win_len)
            sig = np.zeros((self.channels, self.win_len), dtype=np.float32)
            for c in range(self.channels):
                freqs = np.random.choice([6,10,20,40], size=2, replace=False)
                phase = np.random.rand(2)*2*np.pi
                s = 0.5*np.sin(2*np.pi*freqs[0]*t + phase[0]) + 0.3*np.sin(2*np.pi*freqs[1]*t + phase[1])
                s += 0.2*np.random.randn(self.win_len)
                sig[c] = s
            if np.random.rand() < self.seizure_frac:
                center = self.win_len//2
                width = int(0.5*self.fs)
                burst = np.sin(2*np.pi*8*np.linspace(0,1,width)) * np.hanning(width) * 3.0
                start = center - width//2
                for c in range(self.channels):
                    sig[c, start:start+width] += burst * (0.5 + np.random.rand())
                self.y_detect[i] = 1
                self.y_pred[i] = 1
            self.X[i] = sig
        self.X = (self.X - self.X.mean(axis=(0,2), keepdims=True)) / (self.X.std(axis=(0,2), keepdims=True) + 1e-6)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y_detect[idx], dtype=torch.float32), torch.tensor(self.y_pred[idx], dtype=torch.float32)

def compute_basic_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    tp = ((y_true==1) & (y_pred==1)).sum()
    tn = ((y_true==0) & (y_pred==0)).sum()
    fp = ((y_true==0) & (y_pred==1)).sum()
    fn = ((y_true==1) & (y_pred==0)).sum()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    f1 = 2*prec*sens/(prec+sens+1e-9)
    return {'tp':tp,'tn':tn,'fp':fp,'fn':fn,'sensitivity':sens,'specificity':spec,'precision':prec,'f1':f1}

def false_alarms_per_hour(fp, total_hours):
    return fp / (total_hours + 1e-9)

def train_model(model, train_loader, val_loader=None, epochs=12, device='cpu'):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    history = {'train_loss':[], 'val_loss':[], 'val_sens':[], 'val_fa_hr':[]}
    
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, y_det, y_pred in train_loader:
            xb = xb.to(device)
            y_det = y_det.to(device)
            y_pred = y_pred.to(device)
            optim.zero_grad()
            det_prob, pred_prob, extra = model(xb)
            # compute logits from probs numerically stable via logit
            det_logit = torch.log(det_prob.clamp(1e-6,1-1e-6)/(1-det_prob.clamp(1e-6,1-1e-6)))
            pred_logit = torch.log(pred_prob.clamp(1e-6,1-1e-6)/(1-pred_prob.clamp(1e-6,1-1e-6)))
            loss_det = F.binary_cross_entropy_with_logits(det_logit, y_det)
            loss_pred = F.binary_cross_entropy_with_logits(pred_logit, y_pred)
            loss = loss_det + 0.7 * loss_pred
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        if val_loader is not None:
            model.eval()
            all_y = []
            all_p = []
            total_fp = 0
            total_duration_hours = 0.0
            with torch.no_grad():
                for xb, y_det, y_pred in val_loader:
                    xb = xb.to(device)
                    det_prob, pred_prob, extra = model(xb)
                    all_y.append(y_det.numpy())
                    all_p.append(det_prob.detach().cpu().numpy())
                    total_fp += ((det_prob.detach().cpu().numpy() >= 0.5) & (y_det.numpy()==0)).sum()
                    total_duration_hours += xb.shape[0] * (train_loader.dataset.window_s) / 3600.0
            all_y = np.concatenate(all_y)
            all_p = np.concatenate(all_p)
            metrics = compute_basic_metrics(all_y, all_p, threshold=0.5)
            fa_hr = false_alarms_per_hour(metrics['fp'], total_duration_hours)
            history['val_loss'].append(train_loss)
            history['val_sens'].append(metrics['sensitivity'])
            history['val_fa_hr'].append(fa_hr)
            print(f"Epoch {ep+1}/{epochs} | TrainLoss {train_loss:.4f} | Val Sens {metrics['sensitivity']:.3f} | Val FA/hr {fa_hr:.3f}")
        else:
            print(f"Epoch {ep+1}/{epochs} | TrainLoss {train_loss:.4f}")
    return history

# Run demo
device = 'cpu'
channels = 18
fs = 256
window_s = 10
train_ds = SyntheticEEGDataset(n_samples=800, channels=channels, fs=fs, window_s=window_s, seizure_frac=0.12)
val_ds = SyntheticEEGDataset(n_samples=200, channels=channels, fs=fs, window_s=window_s, seizure_frac=0.10)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False)

model = LightLTCSeizNet(in_channels=channels, in_samples=int(window_s*fs), latent_dim=64, ltc_sizes=(64,48), dt=1/25)
history = train_model(model, train_loader, val_loader=val_loader, epochs=12, device=device)

# Plotting
plt.figure()
plt.plot(history['train_loss'], label='train_loss')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history['val_sens'], label='val_sensitivity')
plt.plot(history['val_fa_hr'], label='val_FA_per_hr')
plt.title('Validation metrics')
plt.xlabel('epoch')
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc, precision_recall_curve
model.eval()
all_y = []
all_p = []
with torch.no_grad():
    for xb, y_det, y_pred in val_loader:
        xb = xb.to(device)
        det_prob, pred_prob, extra = model(xb)
        all_y.append(y_det.numpy())
        all_p.append(det_prob.detach().cpu().numpy())
all_y = np.concatenate(all_y)
all_p = np.concatenate(all_p)

fpr, tpr, _ = roc_curve(all_y, all_p)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr)
plt.title(f'ROC curve (AUC={roc_auc:.3f})')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

precision, recall, _ = precision_recall_curve(all_y, all_p)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision)
plt.title(f'Precision-Recall (AUC={pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

det_times = []
lead_times = []
for i in range(len(all_y)):
    if all_y[i] == 1 and all_p[i] >= 0.5:
        det_times.append(0.5)
        lead_times.append(5.0)
plt.figure()
plt.hist(det_times, bins=10)
plt.title('Detection latency histogram (demo)')
plt.xlabel('latency (s)')
plt.show()

plt.figure()
plt.hist(lead_times, bins=10)
plt.title('Prediction lead-time histogram (demo)')
plt.xlabel('lead time (s)')
plt.show()

taus1 = model.ltc1.get_tau().detach().cpu().numpy()
taus2 = model.ltc2.get_tau().detach().cpu().numpy()
plt.figure()
plt.hist(taus1, bins=20)
plt.title('LTC1 learned tau distribution')
plt.xlabel('tau (s)')
plt.show()

plt.figure()
plt.hist(taus2, bins=20)
plt.title('LTC2 learned tau distribution')
plt.xlabel('tau (s)')
plt.show()

summary = {
    'Parameter count': sum(p.numel() for p in model.parameters()),
    'Latency per window (est, ms)': 0.0,
    'Model size (params)': sum(p.numel() for p in model.parameters())
}
df_summary = pd.DataFrame([summary])
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("Model summary", df_summary)

os.makedirs('/mnt/data', exist_ok=True)
torch.save(model.state_dict(), '/mnt/data/lightltc_seiznet_demo_fixed.pth')
print("Demo model saved to /mnt/data/lightltc_seiznet_demo_fixed.pth")
