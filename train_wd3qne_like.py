import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = "/Users/zhihanqin/Desktop/ms_study/bios777/ID3QNE-algorithm"
DATA_DIR = os.path.join(ROOT, "eicu_processed_full")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========= 1. 读数据 =========
states = np.load(os.path.join(DATA_DIR, "states.npy"))
actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
rewards = np.load(os.path.join(DATA_DIR, "rewards.npy"))
next_states = np.load(os.path.join(DATA_DIR, "next_states.npy"))
dones = np.load(os.path.join(DATA_DIR, "dones.npy"))

print("states:", states.shape)
print("actions:", actions.shape)

n_states = states.shape[1]
n_actions = int(actions.max()) + 1

# 标准化
state_mean = states.mean(axis=0, keepdims=True)
state_std = states.std(axis=0, keepdims=True) + 1e-6
states_norm = (states - state_mean) / state_std
next_states_norm = (next_states - state_mean) / state_std

states_t = torch.from_numpy(states_norm).float()
next_states_t = torch.from_numpy(next_states_norm).float()
actions_t = torch.from_numpy(actions).long()
rewards_t = torch.from_numpy(rewards).float()
dones_t = torch.from_numpy(dones).float()

dataset = TensorDataset(states_t, next_states_t, actions_t, rewards_t, dones_t)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

print("Train size:", len(train_set), "Val size:", len(val_set))

# ========= 2. 原 GitHub 的 DistributionalDQN 结构 =========
class DistributionalDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(DistributionalDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state):
        conv_out = self.conv(state)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ========= 3. WD3QNE-like 训练器（保留加权 double Q，去掉 SOFA/next_action） =========
class WD3QNETrainer:
    def __init__(self, state_dim, n_actions, gamma=0.99, tau=0.1):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.Q = DistributionalDQN(state_dim, n_actions).to(device)
        self.Q_target = DistributionalDQN(state_dim, n_actions).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=1e-4)
        self.n_actions = n_actions

    def polyak_update(self):
        # 这里直接 copy（跟原代码一样），也可以做指数平均
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(param.data)

    def compute_loss(self, s, s_next, a, r, d):
        """
        s, s_next: (B, state_dim)
        a: (B,)
        r, d: (B,)
        """
        s = s.to(self.device)
        s_next = s_next.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        batch_size = s.size(0)
        idx = torch.arange(batch_size, device=self.device).long()

        # 当前 Q(s,a)
        Q_pred = self.Q(s)                  # (B, A)
        Q_sa = Q_pred[idx, a]               # (B,)

        # Double Q + WQNE 部分
        with torch.no_grad():
            # 1) 用 online Q 选 next_action
            Q_next_eval = self.Q(s_next)                # (B, A)
            a_eval = Q_next_eval.argmax(dim=1)          # (B,)

            # 2) 用 target Q 计算两个版本的 Q(s',a)
            Q_target_all = self.Q_target(s_next)        # (B, A)
            a_target = Q_target_all.argmax(dim=1)       # (B,)

            Q_eval_next = Q_target_all[idx, a_eval]     # (B,)
            Q_tar_next = Q_target_all[idx, a_target]    # (B,)

            # 3) 用 softmax 权重加权
            probs = torch.softmax(Q_target_all, dim=1)  # (B, A)
            p1 = probs[idx, a_eval]                     # (B,)
            p2 = probs[idx, a_target]                   # (B,)
            mix = (p1 / (p1 + p2 + 1e-8)) * Q_eval_next + \
                  (p2 / (p1 + p2 + 1e-8)) * Q_tar_next  # (B,)

            target = r + self.gamma * (1.0 - d) * mix    # (B,)

        loss = nn.SmoothL1Loss()(Q_sa, target)
        return loss

# ========= 4. 训练循环 =========
trainer = WD3QNETrainer(n_states, n_actions)

def eval_loss():
    trainer.Q.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for s, s_next, a, r, d in val_loader:
            loss = trainer.compute_loss(s, s_next, a, r, d)
            total += loss.item() * s.size(0)
            n += s.size(0)
    return total / max(1, n)

epochs = 30
for epoch in range(1, epochs + 1):
    trainer.Q.train()
    total = 0.0
    n = 0
    for s, s_next, a, r, d in train_loader:
        loss = trainer.compute_loss(s, s_next, a, r, d)
        trainer.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.Q.parameters(), 1.0)
        trainer.optimizer.step()
        total += loss.item() * s.size(0)
        n += s.size(0)

    if epoch % 5 == 0:
        trainer.polyak_update()

    train_loss = total / max(1, n)
    val_loss = eval_loss()
    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

# ========= 5. 保存模型 =========
save_path = os.path.join(MODEL_DIR, "wd3qne_like_sepsis.pt")
torch.save({
    "state_dict": trainer.Q.state_dict(),
    "state_mean": state_mean,
    "state_std": state_std,
    "n_states": n_states,
    "n_actions": n_actions
}, save_path)
print("Saved WD3QNE-like model to", save_path)
