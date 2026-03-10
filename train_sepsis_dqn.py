import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ======== 路径设置 ========
ROOT = "/Users/zhihanqin/Desktop/ms_study/bios777/ID3QNE-algorithm"
DATA_DIR = os.path.join(ROOT, "eicu_processed_full")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======== 1. 读取离线 RL 数据 ========
states = np.load(os.path.join(DATA_DIR, "states.npy"))
actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
rewards = np.load(os.path.join(DATA_DIR, "rewards.npy"))
next_states = np.load(os.path.join(DATA_DIR, "next_states.npy"))
dones = np.load(os.path.join(DATA_DIR, "dones.npy"))

print("states shape:", states.shape)
print("actions shape:", actions.shape)
print("rewards shape:", rewards.shape)

n_states = states.shape[1]
n_actions = int(actions.max()) + 1
print("n_states =", n_states, "n_actions =", n_actions)

# ======== 2. 状态标准化 ========
state_mean = states.mean(axis=0, keepdims=True)
state_std = states.std(axis=0, keepdims=True) + 1e-6

states_norm = (states - state_mean) / state_std
next_states_norm = (next_states - state_mean) / state_std

# 如数据太大，可以先随机采样一部分做 demo
# idx = np.random.choice(len(states_norm), size=min(80000, len(states_norm)), replace=False)
# states_norm = states_norm[idx]
# actions = actions[idx]
# rewards = rewards[idx]
# next_states_norm = next_states_norm[idx]
# dones = dones[idx]

# ======== 3. 构造 DataLoader ========
states_t = torch.from_numpy(states_norm).float()
actions_t = torch.from_numpy(actions).long()
rewards_t = torch.from_numpy(rewards).float()
next_states_t = torch.from_numpy(next_states_norm).float()
dones_t = torch.from_numpy(dones).float()

dataset = TensorDataset(states_t, actions_t, rewards_t, next_states_t, dones_t)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)

print("Train size:", len(train_set), "Val size:", len(val_set))

# ======== 4. Dueling Double DQN 网络 ========

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        feat = self.feature(x)
        value = self.value_head(feat)            # (B,1)
        adv = self.adv_head(feat)                # (B,A)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = value + (adv - adv_mean)             # dueling 合成
        return q

# ======== 5. 初始化网络与优化器 ========
gamma = 0.99
lr = 1e-3
epochs = 30
target_update_interval = 5

policy_net = DuelingDQN(n_states, n_actions).to(device)
target_net = DuelingDQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ======== 6. 验证函数（离线 loss） ========
def evaluate(loader):
    policy_net.eval()
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for s, a, r, s_next, d in loader:
            s = s.to(device)
            a = a.to(device)
            r = r.to(device)
            s_next = s_next.to(device)
            d = d.to(device)

            q = policy_net(s)
            q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)

            q_next_policy = policy_net(s_next)
            next_a = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target = target_net(s_next)
            q_next = q_next_target.gather(1, next_a).squeeze(1)
            target = r + gamma * (1 - d) * q_next

            loss = loss_fn(q_sa, target)
            total_loss += loss.item() * s.size(0)
            total_n += s.size(0)

    return total_loss / max(1, total_n)

# ======== 7. 训练循环（离线 Double DQN） ========
for epoch in range(1, epochs + 1):
    policy_net.train()
    total_loss = 0.0
    total_n = 0

    for s, a, r, s_next, d in train_loader:
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        s_next = s_next.to(device)
        d = d.to(device)

        q = policy_net(s)
        q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            q_next_policy = policy_net(s_next)
            next_a = q_next_policy.argmax(dim=1, keepdim=True)
            q_next_target = target_net(s_next)
            q_next = q_next_target.gather(1, next_a).squeeze(1)
            target = r + gamma * (1 - d) * q_next

        loss = loss_fn(q_sa, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * s.size(0)
        total_n += s.size(0)

    train_loss = total_loss / max(1, total_n)
    val_loss = evaluate(val_loader)

    if epoch % target_update_interval == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

# ======== 8. 保存模型和标准化参数 ========
save_path = os.path.join(MODEL_DIR, "dueling_dqn_sepsis.pt")
torch.save({
    "policy_state_dict": policy_net.state_dict(),
    "state_mean": state_mean,
    "state_std": state_std,
    "n_states": n_states,
    "n_actions": n_actions
}, save_path)

print("Model saved to", save_path)
