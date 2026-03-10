import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

ROOT = "/Users/zhihanqin/Desktop/ms_study/bios777/ID3QNE-algorithm"
DATA_DIR = os.path.join(ROOT, "eicu_processed_full")
MODEL_PATH = os.path.join(ROOT, "models", "wd3qne_like_sepsis.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========= 1. 读数据 =========
states = np.load(os.path.join(DATA_DIR, "states.npy"))
actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
rewards = np.load(os.path.join(DATA_DIR, "rewards.npy"))
next_states = np.load(os.path.join(DATA_DIR, "next_states.npy"))
dones = np.load(os.path.join(DATA_DIR, "dones.npy"))

print("Loaded:", states.shape, actions.shape, rewards.shape, dones.shape)

n_states = states.shape[1]
n_actions = int(actions.max()) + 1

# ========= 2. 定义与训练时一致的网络 =========
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

# ========= 3. 加载模型 =========
ckpt = torch.load(MODEL_PATH, map_location=device)
state_mean = ckpt["state_mean"]
state_std = ckpt["state_std"]

policy_net = DistributionalDQN(ckpt["n_states"], ckpt["n_actions"]).to(device)
policy_net.load_state_dict(ckpt["state_dict"])
policy_net.eval()

# 标准化
states_norm = (states - state_mean) / state_std
states_t = torch.from_numpy(states_norm).float().to(device)

# ========= 4. “do_eval”：agent 动作 vs 临床动作 =========
with torch.no_grad():
    Q_value = policy_net(states_t)                      # (N, A)
    agent_actions = Q_value.argmax(dim=1).cpu().numpy()
    phy_actions = actions                               # clinician行为
    Q_value_pro1 = F.softmax(Q_value, dim=1)
    max_ind = Q_value_pro1.argmax(dim=1)
    idx = torch.arange(Q_value_pro1.size(0), device=device)
    Q_value_pro = Q_value_pro1[idx, max_ind].cpu().numpy()

# 行为一致性
agree = (agent_actions == phy_actions).mean()
print(f"Agent vs clinician action agreement: {agree:.3f}")

# draw Survival vs Expected Return
episodes = []
start = 0
for i, d in enumerate(dones):
    if d == 1:
        episodes.append(slice(start, i + 1))
        start = i + 1
if start < len(dones):
    episodes.append(slice(start, len(dones)))

print("Number of episodes:", len(episodes))

gamma = 0.99
episode_returns = []
episode_survival = []
episode_q_value = []

with torch.no_grad():
    for sl in episodes:
        r_ep = rewards[sl]
        s0 = states_norm[sl.start]

        # 折扣回报
        G = 0.0
        for t, r in enumerate(r_ep):
            G += (gamma ** t) * r
        episode_returns.append(G)

        # 生存标签（终末 reward = ±24）
        R_T = r_ep[-1]
        surv = 1 if R_T > 0 else 0
        episode_survival.append(surv)

        # WD3QNE-like 的 V_hat(s0)
        s0_t = torch.from_numpy(s0).float().unsqueeze(0).to(device)
        q_vals = policy_net(s0_t)
        v_hat = q_vals.max(dim=1)[0].item()
        episode_q_value.append(v_hat)

episode_returns = np.array(episode_returns)
episode_survival = np.array(episode_survival)
episode_q_value = np.array(episode_q_value)

print("Episode returns mean:", episode_returns.mean())
print("Episode survival:", episode_survival.mean())

# ========= 6. 分箱画图 =========
num_bins = 10
percentiles = np.linspace(0, 100, num_bins + 1)
bins = np.percentile(episode_q_value, percentiles)

bin_centers = []
bin_survival = []

for i in range(num_bins):
    low, high = bins[i], bins[i+1]
    if i == num_bins - 1:
        mask = (episode_q_value >= low) & (episode_q_value <= high)
    else:
        mask = (episode_q_value >= low) & (episode_q_value < high)
    if mask.sum() == 0:
        continue
    bin_centers.append(episode_q_value[mask].mean())
    bin_survival.append(episode_survival[mask].mean())

bin_centers = np.array(bin_centers)
bin_survival = np.array(bin_survival)

print("Binned survival (WD3QNE-like):")
for x, y in zip(bin_centers, bin_survival):
    print(f"V≈{x:.2f}, survival≈{y:.2f}")

plt.figure()
plt.plot(bin_centers, bin_survival, marker='o')
plt.xlabel("Expected Return (V(s0) from WD3QNE-like)")
plt.ylabel("Observed Survival Rate")
plt.title("Survival vs Expected Return (WD3QNE-like, eICU)")
plt.grid(True)

out_png = os.path.join(ROOT, "survival_vs_return_wd3qne_like.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Plot saved to", out_png)

# ========= 7. 保存一些 Q值/动作，方便后续分析 =========
np.save(os.path.join(ROOT, "Q_agent_actions.npy"), agent_actions)
np.save(os.path.join(ROOT, "Q_phys_actions.npy"), phy_actions)
np.save(os.path.join(ROOT, "Q_agent_confidence.npy"), Q_value_pro)
