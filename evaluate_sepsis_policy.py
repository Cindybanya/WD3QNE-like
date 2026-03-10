import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ========= 路径 =========
ROOT = "/Users/zhihanqin/Desktop/ms_study/bios777/ID3QNE-algorithm"
DATA_DIR = os.path.join(ROOT, "eicu_processed_full")
MODEL_PATH = os.path.join(ROOT, "models", "dueling_dqn_sepsis.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========= 1. 读取数据 =========
states = np.load(os.path.join(DATA_DIR, "states.npy"))
actions = np.load(os.path.join(DATA_DIR, "actions.npy"))
rewards = np.load(os.path.join(DATA_DIR, "rewards.npy"))
next_states = np.load(os.path.join(DATA_DIR, "next_states.npy"))
dones = np.load(os.path.join(DATA_DIR, "dones.npy"))

print("Loaded:", states.shape, actions.shape, rewards.shape, dones.shape)

n_states = states.shape[1]
n_actions = int(actions.max()) + 1

# ========= 2. 定义 DuelingDQN（要和训练时完全一样） =========
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
        value = self.value_head(feat)
        adv = self.adv_head(feat)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q = value + (adv - adv_mean)
        return q

# ========= 3. 加载模型和标准化参数 =========
ckpt = torch.load(MODEL_PATH, map_location=device)
state_mean = ckpt["state_mean"]
state_std = ckpt["state_std"]

policy_net = DuelingDQN(ckpt["n_states"], ckpt["n_actions"]).to(device)
policy_net.load_state_dict(ckpt["policy_state_dict"])
policy_net.eval()

# 标准化
states_norm = (states - state_mean) / state_std

# ========= 4. 按 episode 重建轨迹 =========
# 我们在 preprocess 里是按 stay 依次 append 的，dones==1 表示一个 episode 结束
episodes = []
start = 0
for i, d in enumerate(dones):
    if d == 1:
        end = i + 1
        episodes.append(slice(start, end))
        start = end
if start < len(dones):
    episodes.append(slice(start, len(dones)))

print("Number of episodes:", len(episodes))

gamma = 0.99
episode_returns = []        # 真正的折扣回报 ∑γ^t r_t
episode_survival = []       # 终末奖励 sign，用来近似生存：R_T>0 → survive
episode_q_value = []        # 我们模型给的 expected return 估计：max_a Q(s0,a)

with torch.no_grad():
    for sl in episodes:
        r_ep = rewards[sl]
        d_ep = dones[sl]
        s0 = states_norm[sl.start]  # episode 第一个 state

        # 1) Monte Carlo 折扣回报
        G = 0.0
        for t, r in enumerate(r_ep):
            G += (gamma ** t) * r
        episode_returns.append(G)

        # 2) survival（我们构造过：终末 reward = +24 (生存) 或 -24 (死亡)）
        R_T = r_ep[-1]
        surv = 1 if R_T > 0 else 0
        episode_survival.append(surv)

        # 3) 模型的 V(s0) ≈ max_a Q(s0,a)
        s0_t = torch.from_numpy(s0).float().unsqueeze(0).to(device)
        q_values = policy_net(s0_t)
        v_hat = q_values.max(dim=1)[0].item()
        episode_q_value.append(v_hat)

episode_returns = np.array(episode_returns)
episode_survival = np.array(episode_survival)
episode_q_value = np.array(episode_q_value)

print("Episode returns: mean =", episode_returns.mean())
print("Episode survival rate:", episode_survival.mean())

# ========= 5. 画一个 “Expected return vs Survival rate”（类似 fig.5） =========
# 做一个简单的分箱：按模型给的 V(s0)（episode_q_value）分成 10 组
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

print("Binned survival (for plotting):")
for x, y in zip(bin_centers, bin_survival):
    print(f"V≈{x:.2f}, survival≈{y:.2f}")

# 做一个简单的线图
plt.figure()
plt.plot(bin_centers, bin_survival, marker='o')
plt.xlabel("Expected Return (approx. V(s0) from DQN)")
plt.ylabel("Observed Survival Rate")
plt.title("Survival vs Expected Return (DQN, eICU)")
plt.grid(True)

out_png = os.path.join(ROOT, "survival_vs_return_dqn.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
print("Plot saved to", out_png)
