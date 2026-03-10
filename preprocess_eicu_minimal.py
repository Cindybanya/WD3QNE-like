import os
import numpy as np
import pandas as pd

# ====== PATH SETTINGS ======
ROOT = "/Users/zhihanqin/Desktop/ms_study/bios777/ID3QNE-algorithm"
RAW = os.path.join(ROOT, "eicu_raw")
OUT = os.path.join(ROOT, "eicu_processed_full")
os.makedirs(OUT, exist_ok=True)

def load(name):
    print(f"Loading {name} ...")
    return pd.read_csv(os.path.join(RAW, name), compression="gzip", low_memory=False)

# ====== LOAD TABLES ======
patient = load("patient.csv.gz")
adm = load("admissionDx.csv.gz")
inf = load("infusionDrug.csv.gz")
io = load("intakeOutput.csv.gz")
lab = load("lab.csv.gz")
vp = load("vitalPeriodic.csv.gz")
va = load("vitalAperiodic.csv.gz")

print("loaded")

# ====== STEP 1: SIMPLE SEPSIS COHORT ======
# admissionDx.csv.gz 里面的 admitdxname 有 “Sepsis”, “infection”
sepsis_ids = adm[adm["admitdxname"].str.contains("sepsis|Sepsis|infection|Infection",
                                                 case=False, na=False)]["patientunitstayid"].unique()

cohort = patient[patient["patientunitstayid"].isin(sepsis_ids)].copy()
stay_ids = set(cohort["patientunitstayid"])


print("Selected sepsis stays:", len(stay_ids))

# filter all tables
inf = inf[inf["patientunitstayid"].isin(stay_ids)]
io = io[io["patientunitstayid"].isin(stay_ids)]
lab = lab[lab["patientunitstayid"].isin(stay_ids)]
vp  = vp[vp["patientunitstayid"].isin(stay_ids)]
va  = va[va["patientunitstayid"].isin(stay_ids)]

# ====== STEP 2: MAKE 4-HOUR BINS ======
def bin_by_4h(df, offset_col):
    df = df.copy()
    df["bin"] = (df[offset_col] / 60 // 4).astype(int)
    return df

vp = bin_by_4h(vp, "observationoffset")
va = bin_by_4h(va, "observationoffset")
lab = bin_by_4h(lab, "labresultoffset")
inf = bin_by_4h(inf, "infusionoffset")
io  = bin_by_4h(io, "intakeoutputoffset")

# ====== STEP 3: BUILD STATE FEATURES ======
# vitals periodic
state_vitals = (
    vp.groupby(["patientunitstayid", "bin"])
      .agg({
          "temperature": "mean",
          "sao2": "mean",
          "heartrate": "mean",
          "respiration": "mean",
          "systemicsystolic": "mean",
          "systemicdiastolic": "mean",
          "systemicmean": "mean"
      })
      .reset_index()
)

# labs: pick several common ones
important_labs = ["Creatinine", "BUN", "Platelet Count", "Bilirubin"]
lab_sub = lab[lab["labname"].isin(important_labs)].copy()

state_lab = (
    lab_sub.groupby(["patientunitstayid", "bin", "labname"])["labresult"]
           .mean()
           .reset_index()
           .pivot(index=["patientunitstayid", "bin"],
                  columns="labname",
                  values="labresult")
           .reset_index()
)

# merge vitals + labs
state = state_vitals.merge(state_lab, on=["patientunitstayid", "bin"], how="left")

# demographics
demo = cohort[["patientunitstayid", "age", "gender"]].drop_duplicates()
state = state.merge(demo, on="patientunitstayid", how="left")

# ====== CLEAN STATE FEATURES ======

# 1) gender：Male=1, Female/other=0
state["gender"] = (state["gender"] == "Male").astype(float)

# 2) state 特征强制变成 float
for col in state.columns:
    if col not in ["patientunitstayid", "bin"]:
        state[col] = pd.to_numeric(state[col], errors="coerce")

# 3) 填补缺失值
for col in state.columns:
    if col not in ["patientunitstayid", "bin"]:
        state[col] = state[col].fillna(state[col].mean())

state = state.sort_values(["patientunitstayid", "bin"])
state_cols = [c for c in state.columns if c not in ["patientunitstayid", "bin"]]

print("State feature columns:", state_cols)

# ====== STEP 4: BUILD ACTIONS (IV + VASOPRESSORS) ======
# IV: intake ml = intaketotal
iv_agg = (
    io.groupby(["patientunitstayid", "bin"])["intaketotal"]
      .sum()
      .reset_index()
      .rename(columns={"intaketotal": "iv_ml"})
)

# vasopressors: drugname contains these
pressors = ["Norepinephrine", "Vasopressin", "Epinephrine", "Phenylephrine"]
inf["is_vaso"] = inf["drugname"].str.contains("|".join(pressors), case=False, na=False)

# 把 drugrate 转成数值，非数字变成 NaN
inf["drugrate"] = pd.to_numeric(inf["drugrate"], errors="coerce")

vaso_agg = (
    inf[inf["is_vaso"]]
    .groupby(["patientunitstayid", "bin"])["drugrate"]
    .mean()
    .reset_index()
    .rename(columns={"drugrate": "vaso_rate"})
)

# 没有给升压药的时间段，vaso_rate 设成 0
vaso_agg["vaso_rate"] = vaso_agg["vaso_rate"].fillna(0.0)

# combine
action_df = state[["patientunitstayid", "bin"]].copy()
action_df = action_df.merge(iv_agg, on=["patientunitstayid", "bin"], how="left")
action_df = action_df.merge(vaso_agg, on=["patientunitstayid", "bin"], how="left")

action_df["iv_ml"] = action_df["iv_ml"].fillna(0)
action_df["vaso_rate"] = action_df["vaso_rate"].fillna(0)

# 5×5 discretization
action_df["IV_bin"] = pd.qcut(action_df["iv_ml"].rank(method="first"),
                              5, labels=False, duplicates="drop").fillna(0).astype(int)

action_df["Vaso_bin"] = pd.qcut(action_df["vaso_rate"].rank(method="first"),
                                5, labels=False, duplicates="drop").fillna(0).astype(int)

action_df["action"] = action_df["IV_bin"] * 5 + action_df["Vaso_bin"]

# ====== STEP 5: REWARDS ======
# simplified terminal reward (±24)
outcome = cohort[["patientunitstayid", "unitdischargestatus"]].copy()
outcome["dead"] = (outcome["unitdischargestatus"] == "Expired").astype(int)

merged = state.merge(action_df, on=["patientunitstayid", "bin"], how="left")
merged = merged.merge(outcome, on="patientunitstayid", how="left")

merged["max_bin"] = merged.groupby("patientunitstayid")["bin"].transform("max")
merged["done"] = (merged["bin"] == merged["max_bin"]).astype(int)

merged["reward"] = 0.0

mask = (merged["done"] == 1)
merged.loc[mask, "reward"] = np.where(
    merged.loc[mask, "dead"] == 1,
    -24.0,
    24.0
)

# ====== STEP 6: BUILD RL TUPLES ======
records = []

for sid, df_sid in merged.groupby("patientunitstayid"):
    df_sid = df_sid.sort_values("bin").reset_index(drop=True)
    for i in range(len(df_sid)):
        s = df_sid.loc[i, state_cols].values.astype(float)
        a = int(df_sid.loc[i, "action"])
        r = float(df_sid.loc[i, "reward"])
        d = int(df_sid.loc[i, "done"])

        if i < len(df_sid)-1:
            s_next = df_sid.loc[i+1, state_cols].values.astype(float)
        else:
            s_next = s.copy()

        records.append((s, a, r, s_next, d))

print("Total transitions:", len(records))

# unpack
states, actions, rewards, next_states, dones = [], [], [], [], []
for s, a, r, sn, d in records:
    states.append(s)
    actions.append(a)
    rewards.append(r)
    next_states.append(sn)
    dones.append(d)

np.save(os.path.join(OUT, "states.npy"), np.stack(states))
np.save(os.path.join(OUT, "actions.npy"), np.array(actions))
np.save(os.path.join(OUT, "rewards.npy"), np.array(rewards))
np.save(os.path.join(OUT, "next_states.npy"), np.stack(next_states))
np.save(os.path.join(OUT, "dones.npy"), np.array(dones))

print("Saved RL data to", OUT)
