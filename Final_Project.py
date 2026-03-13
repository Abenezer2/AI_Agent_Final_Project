#!/usr/bin/env python3
"""
=============================================================================
THE CONTAGION OF CONFLICT v3 (A100 Edition)
How Toxicity Spreads Through Reddit Conversations
=============================================================================

Key upgrades from v2:
  - 15 subreddits (more statistical power + diversity)
  - Context-only GAT ablation (masks own text, tests true contagion)
  - Parent text embedding + author toxicity rate as node features
  - Best-checkpoint reload before test evaluation
  - 3 random seeds with mean +/- std reporting
  - Toxic-class precision/recall/F1 + PR-AUC
  - Data-driven figures only (no generic pipeline/architecture boxes)

Usage:
  DRY RUN:   python toxicity_contagion_v3.py --dry-run
  FULL RUN:  export OPENAI_API_KEY="sk-..." && python toxicity_contagion_v3.py
=============================================================================
"""

import os, sys, json, time, pickle, random, argparse, warnings, copy
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    def __init__(self, dry_run=False, api_key=""):
        self.DRY_RUN = dry_run
        self.DATA_DIR = "data"
        self.RESULTS_DIR = "results"
        self.FIGURES_DIR = "figures"
        self.MODELS_DIR = "models"
        for d in [self.DATA_DIR, self.RESULTS_DIR, self.FIGURES_DIR, self.MODELS_DIR]:
            os.makedirs(d, exist_ok=True)

        self.FALLBACK_CORPUS = "reddit-corpus-small"
        self.NUM_SUBREDDITS = 15
        self.MAX_COMMENTS_PER_SUBREDDIT = 50_000
        self.MIN_CONVERSATION_DEPTH = 3
        self.MIN_CONVERSATION_SIZE = 4

        self.TOXICITY_MODEL_NAME = "s-nlp/roberta_toxicity_classifier"
        self.TOXICITY_BATCH_SIZE = 64
        self.TOXICITY_THRESHOLD = 0.5
        self.EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
        self.EMBEDDING_BATCH_SIZE = 64
        self.EMBEDDING_DIM = 768
        self.DEVICE = "cuda" if torch.cuda.is_available() and not dry_run else "cpu"

        self.GAT_HIDDEN_DIM = 64
        self.GAT_HEADS = 4
        self.GAT_EPOCHS = 60
        self.GAT_LR = 0.001
        self.GAT_DROPOUT = 0.3
        self.GAT_TRAIN_RATIO = 0.7
        self.GAT_VAL_RATIO = 0.15
        self.GAT_SEEDS = [42, 123, 456]

        self.OPENAI_API_KEY = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.OPENAI_MODEL = "gpt-4o-mini"
        self.MAX_THREAD_LEN_FOR_LLM = 40
        self.NUM_NORM_PROFILE_SAMPLES = 100
        self.NUM_QUALITATIVE_CASES = 60
        self.NUM_COUNTERFACTUAL_CASES = 15
        self.MIN_COMMENTS_FOR_USER_PROFILE = 5
        self.FIGURE_DPI = 300
        self.RANDOM_SEED = 42

        if dry_run:
            self.NUM_SUBREDDITS = 3
            self.MAX_COMMENTS_PER_SUBREDDIT = 200
            self.MIN_CONVERSATION_DEPTH = 2
            self.MIN_CONVERSATION_SIZE = 3
            self.TOXICITY_BATCH_SIZE = 8
            self.EMBEDDING_BATCH_SIZE = 16
            self.EMBEDDING_DIM = 32
            self.GAT_HIDDEN_DIM = 8
            self.GAT_HEADS = 2
            self.GAT_EPOCHS = 3
            self.GAT_SEEDS = [42]
            self.NUM_NORM_PROFILE_SAMPLES = 10
            self.NUM_QUALITATIVE_CASES = 6
            self.NUM_COUNTERFACTUAL_CASES = 2
            self.FIGURE_DPI = 100

COLORS = {
    "toxic": "#D32F2F", "clean": "#388E3C", "borderline": "#FFC107",
    "abl_full": "#2E7D32", "abl_context": "#1565C0", "abl_text": "#1976D2",
    "abl_tabular": "#7B1FA2", "abl_nograph": "#F57C00",
}
SUB_CMAP = plt.cm.get_cmap("tab20")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def step_download_data(cfg):
    print("\n" + "=" * 70)
    print(f"STEP 1: LOADING {cfg.NUM_SUBREDDITS} SUBREDDITS")
    print("=" * 70)
    from convokit import Corpus, download
    corpus = Corpus(filename=download(cfg.FALLBACK_CORPUS))
    records = _extract_utterances(corpus)
    df = pd.DataFrame(records)
    sub_counts = df["subreddit"].value_counts()
    selected = sub_counts[sub_counts >= 200].head(cfg.NUM_SUBREDDITS).index.tolist()
    print(f"Selected {len(selected)} subreddits")
    subreddit_dfs = {}
    for sub in selected:
        sub_all = df[df["subreddit"] == sub]
        # Sample WHOLE conversations to preserve reply tree integrity
        conv_ids = sub_all["conversation_id"].unique()
        np.random.seed(cfg.RANDOM_SEED)
        np.random.shuffle(conv_ids)
        keep_ids, total = [], 0
        for cid in conv_ids:
            n = (sub_all["conversation_id"] == cid).sum()
            if total + n > cfg.MAX_COMMENTS_PER_SUBREDDIT:
                break
            keep_ids.append(cid); total += n
        sub_df = sub_all[sub_all["conversation_id"].isin(keep_ids)].copy()
        sub_df.to_parquet(os.path.join(cfg.DATA_DIR, f"{sub}.parquet"), index=False)
        subreddit_dfs[sub] = sub_df
        print(f"  r/{sub}: {len(sub_df)} comments, {sub_df['conversation_id'].nunique()} convos (whole conversations)")
    return list(subreddit_dfs.keys())

def _extract_utterances(corpus):
    records = []
    for utt in corpus.iter_utterances():
        text = utt.text
        if not text or text.strip() == "" or text in ("[deleted]", "[removed]"):
            continue
        try: score = utt.meta.get("score", 0) if utt.meta else 0
        except: score = 0
        try: subreddit = utt.meta.get("subreddit", "unknown") if utt.meta else "unknown"
        except: subreddit = "unknown"
        records.append({"id": utt.id, "text": text,
                        "speaker_id": utt.speaker.id if utt.speaker else "unknown",
                        "reply_to": utt.reply_to, "timestamp": utt.timestamp or 0,
                        "score": score, "conversation_id": utt.conversation_id,
                        "subreddit": subreddit})
    return records

# =============================================================================
# STEP 2: TOXICITY SCORING
# =============================================================================

def step_toxicity_scoring(cfg, subreddit_names):
    print("\n" + "=" * 70)
    print("STEP 2: TOXICITY SCORING")
    print("=" * 70)
    if cfg.DRY_RUN:
        for sub in subreddit_names:
            path = os.path.join(cfg.DATA_DIR, f"{sub}.parquet")
            df = pd.read_parquet(path)
            np.random.seed(cfg.RANDOM_SEED)
            df["toxicity_score"] = np.random.beta(0.5, 2.0, size=len(df))
            df["is_toxic"] = df["toxicity_score"] >= cfg.TOXICITY_THRESHOLD
            df.to_parquet(path, index=False)
        return
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    tokenizer = RobertaTokenizer.from_pretrained(cfg.TOXICITY_MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(cfg.TOXICITY_MODEL_NAME).to(cfg.DEVICE).eval()
    toxic_idx = 1
    if hasattr(model.config, "id2label"):
        for idx, label in model.config.id2label.items():
            if "toxic" in str(label).lower(): toxic_idx = int(idx); break
    print(f"  Toxic class index: {toxic_idx}")
    for sub in subreddit_names:
        path = os.path.join(cfg.DATA_DIR, f"{sub}.parquet")
        df = pd.read_parquet(path); texts = df["text"].tolist(); scores = []
        for i in tqdm(range(0, len(texts), cfg.TOXICITY_BATCH_SIZE), desc=f"  r/{sub}"):
            batch = [t[:1000] for t in texts[i:i+cfg.TOXICITY_BATCH_SIZE]]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(cfg.DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=1)
            scores.extend(probs[:, toxic_idx].cpu().numpy().tolist())
        df["toxicity_score"] = scores
        df["is_toxic"] = df["toxicity_score"] >= cfg.TOXICITY_THRESHOLD
        df.to_parquet(path, index=False)
        print(f"    toxic={df['is_toxic'].mean()*100:.1f}%")
    del model
    if cfg.DEVICE == "cuda": torch.cuda.empty_cache()

# =============================================================================
# STEP 3: GRAPH CONSTRUCTION (parent embedding + author toxicity)
# =============================================================================

def step_build_graphs(cfg, subreddit_names):
    print("\n" + "=" * 70)
    print("STEP 3: GRAPH CONSTRUCTION")
    print("=" * 70)
    from torch_geometric.data import Data

    if cfg.DRY_RUN:
        embed_fn = lambda texts: np.random.randn(len(texts), cfg.EMBEDDING_DIM).astype(np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(cfg.EMBEDDING_MODEL_NAME, device=cfg.DEVICE)
        embed_fn = lambda texts: embed_model.encode([t[:500] for t in texts], batch_size=cfg.EMBEDDING_BATCH_SIZE, show_progress_bar=True)

    # Prior-only author toxicity (no future leakage)
    print("  Computing prior-only author toxicity rates...")
    all_dfs = [pd.read_parquet(os.path.join(cfg.DATA_DIR, f"{s}.parquet")) for s in subreddit_names]
    gdf = pd.concat(all_dfs, ignore_index=True).sort_values("timestamp")
    # For each comment, compute the author's toxicity rate from EARLIER comments only
    author_prior_tox = {}
    author_history = defaultdict(list)  # speaker_id -> list of (timestamp, is_toxic)
    for _, row in gdf.iterrows():
        sid = row["speaker_id"]
        prior = author_history[sid]
        if len(prior) >= 3:  # require at least 3 prior comments for stable estimate
            author_prior_tox[row["id"]] = float(np.mean([t for _, t in prior]))
        else:
            author_prior_tox[row["id"]] = 0.0  # insufficient history
        author_history[sid].append((row["timestamp"], float(row["is_toxic"])))
    del gdf

    all_graph_data = {}
    for sub in subreddit_names:
        path = os.path.join(cfg.DATA_DIR, f"{sub}.parquet")
        df = pd.read_parquet(path)
        print(f"\n  r/{sub} ({len(df)} comments)...")
        embeddings = embed_fn(df["text"].tolist())
        emb_dim = embeddings.shape[1]

        df["score_norm"] = (df["score"] - df["score"].mean()) / (df["score"].std() + 1e-8)
        id_to_idx = {row["id"]: i for i, (_, row) in enumerate(df.iterrows())}

        # Thread depth
        depths = {}
        for _, row in df.iterrows():
            depth, current, visited = 0, row["reply_to"], set()
            while current and current in id_to_idx and current not in visited:
                visited.add(current); depth += 1
                current = df.iloc[id_to_idx[current]]["reply_to"]
            depths[row["id"]] = depth
        df["thread_depth"] = df["id"].map(depths)
        # Prior-only author toxicity (computed from earlier comments only, no leakage)
        df["author_tox_rate"] = df["id"].map(lambda cid: author_prior_tox.get(cid, 0.0))
        df.to_parquet(path, index=False)

        graphs = []; skipped = 0
        for conv_id, conv_df in df.groupby("conversation_id"):
            conv_df = conv_df.reset_index(drop=True)
            if len(conv_df) < cfg.MIN_CONVERSATION_SIZE or conv_df["thread_depth"].max() < cfg.MIN_CONVERSATION_DEPTH:
                skipped += 1; continue
            conv_ids = conv_df["id"].tolist()
            local_map = {cid: i for i, cid in enumerate(conv_ids)}
            conv_set = set(conv_ids)

            src, dst = [], []
            for _, row in conv_df.iterrows():
                if row["reply_to"] and row["reply_to"] in conv_set:
                    p, c = local_map[row["reply_to"]], local_map[row["id"]]
                    src.extend([p, c]); dst.extend([c, p])
            if not src: skipped += 1; continue

            gidx = [id_to_idx[cid] for cid in conv_ids]
            own_embs = embeddings[gidx]
            parent_embs = np.zeros_like(own_embs)
            for li, (_, row) in enumerate(conv_df.iterrows()):
                if row["reply_to"] and row["reply_to"] in id_to_idx:
                    parent_embs[li] = embeddings[id_to_idx[row["reply_to"]]]

            score_n = conv_df["score_norm"].values.astype(np.float32).reshape(-1, 1)
            ts = conv_df["timestamp"].values.astype(np.float64)
            ts_n = ((ts - ts.min()) / (ts.max() - ts.min() + 1e-8)).astype(np.float32).reshape(-1, 1)
            depth_n = conv_df["thread_depth"].values.astype(np.float32).reshape(-1, 1)
            atox = conv_df["author_tox_rate"].values.astype(np.float32).reshape(-1, 1)

            # Layout: [own_text | parent_text | score | ts | depth | author_tox]
            feats = np.concatenate([own_embs, parent_embs, score_n, ts_n, depth_n, atox], axis=1)
            labels = conv_df["is_toxic"].astype(int).values

            data = Data(x=torch.tensor(feats, dtype=torch.float),
                        edge_index=torch.tensor([src, dst], dtype=torch.long),
                        y=torch.tensor(labels, dtype=torch.long))
            data.conversation_id = conv_id
            data.subreddit = sub
            data.comment_ids = conv_ids
            data.emb_dim = emb_dim
            graphs.append(data)

        torch.save(graphs, os.path.join(cfg.DATA_DIR, f"{sub}_graphs.pt"))
        all_graph_data[sub] = graphs
        print(f"    Graphs: {len(graphs)} (skipped {skipped})")
        if graphs: print(f"    Avg nodes: {np.mean([g.x.shape[0] for g in graphs]):.1f}, Feat dim: {graphs[0].x.shape[1]}")

        # User network
        ug = nx.Graph()
        for _, row in df.iterrows():
            if row["reply_to"] and row["reply_to"] in id_to_idx:
                ps = df.iloc[id_to_idx[row["reply_to"]]]["speaker_id"]; cs = row["speaker_id"]
                if ps != cs:
                    if ug.has_edge(ps, cs): ug[ps][cs]["weight"] += 1
                    else: ug.add_edge(ps, cs, weight=1)
        utox = df.groupby("speaker_id")["toxicity_score"].mean().to_dict()
        nx.set_node_attributes(ug, utox, "avg_toxicity")
        pickle.dump(ug, open(os.path.join(cfg.DATA_DIR, f"{sub}_user_network.pkl"), "wb"))

    if not cfg.DRY_RUN:
        del embed_model
        if cfg.DEVICE == "cuda": torch.cuda.empty_cache()
    return all_graph_data

# =============================================================================
# STEP 4: GAT (context-only ablation, best checkpoint, multi-seed)
# =============================================================================

class ToxicityGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, dropout):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout_rate = dropout
    def forward(self, x, edge_index):
        x = F.dropout(F.elu(self.conv1(x, edge_index)), p=self.dropout_rate, training=self.training)
        return self.classifier(F.elu(self.conv2(x, edge_index)))

def step_train_gat(cfg, subreddit_names):
    print("\n" + "=" * 70)
    print("STEP 4: GAT TRAINING (context-only, multi-seed)")
    print("=" * 70)
    from torch_geometric.loader import DataLoader

    all_graphs = []
    for sub in subreddit_names:
        p = os.path.join(cfg.DATA_DIR, f"{sub}_graphs.pt")
        if os.path.exists(p): all_graphs.extend(torch.load(p, weights_only=False))
    if not all_graphs: print("No graphs!"); return {}

    in_dim = all_graphs[0].x.shape[1]
    emb_dim = all_graphs[0].emb_dim if hasattr(all_graphs[0], "emb_dim") else (in_dim - 4) // 2
    own_end = emb_dim
    par_end = emb_dim * 2
    tab_start = par_end
    print(f"Graphs: {len(all_graphs)}, dim: {in_dim}, own[0:{own_end}], parent[{own_end}:{par_end}], tab[{tab_start}:]")

    ablation_configs = {
        "full":         {"mask_own": False, "mask_par": False, "mask_tab": False, "mask_score": False, "self_loops": False},
        "context_only": {"mask_own": True,  "mask_par": False, "mask_tab": False, "mask_score": True,  "self_loops": False},
        "all_text":     {"mask_own": False, "mask_par": False, "mask_tab": True,  "mask_score": False, "self_loops": False},
        "no_graph":     {"mask_own": False, "mask_par": False, "mask_tab": False, "mask_score": False, "self_loops": True},
        "tabular_only": {"mask_own": True,  "mask_par": True,  "mask_tab": False, "mask_score": False, "self_loops": False},
    }

    all_seed_results = {v: [] for v in ablation_configs}
    training_curves = {}

    for si, seed in enumerate(cfg.GAT_SEEDS):
        print(f"\n--- SEED {seed} ({si+1}/{len(cfg.GAT_SEEDS)}) ---")
        set_seed(seed)
        idx = list(range(len(all_graphs))); random.shuffle(idx)
        nt = int(len(idx)*cfg.GAT_TRAIN_RATIO); nv = int(len(idx)*cfg.GAT_VAL_RATIO)
        train_g = [all_graphs[i] for i in idx[:nt]]
        val_g = [all_graphs[i] for i in idx[nt:nt+nv]]
        test_g = [all_graphs[i] for i in idx[nt+nv:]]

        all_lab = torch.cat([g.y for g in train_g])
        ntox = (all_lab==1).sum().item(); ntot = len(all_lab)
        cw = torch.tensor([1.0,1.0]) if ntox==0 or ntox==ntot else torch.tensor([ntox/ntot, (ntot-ntox)/ntot])
        cw = cw.to(cfg.DEVICE)

        for vn, vc in ablation_configs.items():
            model = ToxicityGAT(in_dim, cfg.GAT_HIDDEN_DIM, cfg.GAT_HEADS, cfg.GAT_DROPOUT).to(cfg.DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.GAT_LR)
            crit = nn.CrossEntropyLoss(weight=cw)

            def prep(gl):
                out = []
                for g in gl:
                    g2 = g.clone(); x = g2.x.clone()
                    if vc["mask_own"]: x[:, :own_end] = 0.0
                    if vc["mask_par"]: x[:, own_end:par_end] = 0.0
                    if vc["mask_tab"]: x[:, tab_start:] = 0.0
                    if vc["mask_score"]: x[:, tab_start] = 0.0  # zero target's karma (post-hoc info)
                    g2.x = x
                    if vc["self_loops"]:
                        n = g2.x.shape[0]
                        g2.edge_index = torch.stack([torch.arange(n), torch.arange(n)])
                    out.append(g2)
                return out

            tl = DataLoader(prep(train_g), batch_size=32, shuffle=True)
            vl = DataLoader(prep(val_g), batch_size=32)
            tel = DataLoader(prep(test_g), batch_size=32)

            best_f1, best_state = 0, None
            e_losses = []
            for epoch in range(cfg.GAT_EPOCHS):
                model.train(); tloss = 0
                for b in tl:
                    b = b.to(cfg.DEVICE); opt.zero_grad()
                    loss = crit(model(b.x, b.edge_index), b.y); loss.backward(); opt.step()
                    tloss += loss.item()
                e_losses.append(tloss/max(1,len(tl)))

                model.eval(); vpr, vlb = [], []
                with torch.no_grad():
                    for b in vl:
                        b = b.to(cfg.DEVICE)
                        vpr.extend(model(b.x, b.edge_index).argmax(dim=1).cpu().numpy())
                        vlb.extend(b.y.cpu().numpy())
                vf = f1_score(vlb, vpr, average="macro", zero_division=0)
                if vf > best_f1: best_f1 = vf; best_state = copy.deepcopy(model.state_dict())

            if si == 0: training_curves[vn] = e_losses
            if best_state: model.load_state_dict(best_state)

            model.eval(); tpr, tlb, tpb = [], [], []
            with torch.no_grad():
                for b in tel:
                    b = b.to(cfg.DEVICE); out = model(b.x, b.edge_index)
                    tpb.extend(torch.softmax(out, dim=1)[:,1].cpu().numpy())
                    tpr.extend(out.argmax(dim=1).cpu().numpy())
                    tlb.extend(b.y.cpu().numpy())
            tlb, tpr, tpb = np.array(tlb), np.array(tpr), np.array(tpb)
            has2 = len(np.unique(tlb)) > 1

            m = {"accuracy": float(accuracy_score(tlb, tpr)),
                 "f1_macro": float(f1_score(tlb, tpr, average="macro", zero_division=0)),
                 "f1_toxic": float(f1_score(tlb, tpr, pos_label=1, zero_division=0)),
                 "prec_toxic": float(precision_score(tlb, tpr, pos_label=1, zero_division=0)),
                 "rec_toxic": float(recall_score(tlb, tpr, pos_label=1, zero_division=0)),
                 "auc": float(roc_auc_score(tlb, tpb) if has2 else 0.5),
                 "pr_auc": float(average_precision_score(tlb, tpb) if has2 else 0.0)}
            all_seed_results[vn].append(m)

            if si == 0:
                print(f"  {vn:<15} acc={m['accuracy']:.3f} f1m={m['f1_macro']:.3f} f1t={m['f1_toxic']:.3f} auc={m['auc']:.3f} prauc={m['pr_auc']:.3f}")
                if has2:
                    fpr, tpr_c, _ = roc_curve(tlb, tpb)
                    all_seed_results[vn][0]["roc_fpr"] = fpr.tolist()
                    all_seed_results[vn][0]["roc_tpr"] = tpr_c.tolist()
                if vn == "full":
                    subs_list = []
                    for g in test_g: subs_list.extend([g.subreddit]*g.x.shape[0] if hasattr(g,"subreddit") else [])
                    sa = np.array(subs_list[:len(tlb)])
                    psa = {s: float(accuracy_score(tlb[sa==s], tpr[sa==s])) for s in subreddit_names if (sa==s).sum()>0}
                    tp_dict = {"labels": tlb.tolist(), "predictions": tpr.tolist(), "probabilities": tpb.tolist(),
                               "per_subreddit_accuracy": psa, "confusion_matrix": confusion_matrix(tlb, tpr).tolist()}
                    if has2:
                        fpr2, tpr2, _ = roc_curve(tlb, tpb)
                        prc, rec, _ = precision_recall_curve(tlb, tpb)
                        tp_dict["roc"] = {"fpr": fpr2.tolist(), "tpr": tpr2.tolist()}
                        tp_dict["pr_curve"] = {"precision": prc.tolist(), "recall": rec.tolist()}
                    json.dump(tp_dict, open(os.path.join(cfg.RESULTS_DIR, "test_predictions.json"), "w"), indent=2)
                    if best_state: torch.save(best_state, os.path.join(cfg.MODELS_DIR, "gat_full.pt"))

    # Aggregate
    ablation_results = {}
    mnames = ["accuracy","f1_macro","f1_toxic","prec_toxic","rec_toxic","auc","pr_auc"]
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    for vn in ablation_configs:
        sm = all_seed_results[vn]; agg = {}
        for mn in mnames:
            vals = [s[mn] for s in sm]
            agg[mn] = float(np.mean(vals)); agg[f"{mn}_std"] = float(np.std(vals))
        if sm and "roc_fpr" in sm[0]: agg["roc_fpr"] = sm[0]["roc_fpr"]; agg["roc_tpr"] = sm[0]["roc_tpr"]
        ablation_results[vn] = agg
        print(f"  {vn:<15} " + " ".join(f"{mn}={agg[mn]:.3f}+/-{agg[f'{mn}_std']:.3f}" for mn in mnames[:4]))

    ablation_results["_training_curves"] = training_curves
    json.dump(ablation_results, open(os.path.join(cfg.RESULTS_DIR, "ablation_results.json"), "w"), indent=2)
    return ablation_results

# =============================================================================
# STEP 5: CONTAGION ANALYSIS
# =============================================================================

def step_contagion_analysis(cfg, subreddit_names):
    print("\n" + "=" * 70)
    print("STEP 5: CONTAGION ANALYSIS")
    print("=" * 70)
    all_records = []
    for sub in subreddit_names:
        df = pd.read_parquet(os.path.join(cfg.DATA_DIR, f"{sub}.parquet"))
        id2r = {r["id"]: r.to_dict() for _, r in df.iterrows()}
        ch = defaultdict(list)
        for _, r in df.iterrows():
            if r["reply_to"] and r["reply_to"] in id2r: ch[r["reply_to"]].append(r["id"])
        for _, row in df.iterrows():
            if not row["reply_to"] or row["reply_to"] not in id2r: continue
            pr = id2r[row["reply_to"]]; pt = bool(pr["is_toxic"])
            prior_sibs = [s for s in ch.get(row["reply_to"],[]) if s!=row["id"] and id2r[s]["timestamp"]<row["timestamp"]]
            st = [bool(id2r[s]["is_toxic"]) for s in prior_sibs]
            an = [pt] + st; frac = float(np.mean(an))
            tu = set()
            if pt: tu.add(pr["speaker_id"])
            for s in prior_sibs:
                if id2r[s]["is_toxic"]: tu.add(id2r[s]["speaker_id"])
            gp = pr.get("reply_to")
            gpt = bool(id2r[gp]["is_toxic"]) if gp and gp in id2r else False
            all_records.append({"comment_id":row["id"],"subreddit":sub,"is_toxic":bool(row["is_toxic"]),
                                "parent_toxic":pt,"grandparent_toxic":gpt,"frac_toxic_neighbors":frac,
                                "num_distinct_toxic_users":len(tu),"num_prior_siblings":len(prior_sibs),
                                "thread_depth":int(row.get("thread_depth",0)),"score_norm":float(row.get("score_norm",0))})

    fdf = pd.DataFrame(all_records)
    fdf.to_parquet(os.path.join(cfg.RESULTS_DIR, "contagion_features.parquet"), index=False)
    print(f"Total: {len(fdf)}")

    # Curves
    curves = {}; be = np.linspace(0,1,11); bc = ((be[:-1]+be[1:])/2).tolist()
    for sub in subreddit_names + ["combined"]:
        sd = fdf if sub=="combined" else fdf[fdf["subreddit"]==sub]
        if sd.empty: continue
        probs,counts,clo,chi = [],[],[],[]
        for i in range(len(be)-1):
            if i < len(be)-2: mask = (sd["frac_toxic_neighbors"]>=be[i])&(sd["frac_toxic_neighbors"]<be[i+1])
            else: mask = (sd["frac_toxic_neighbors"]>=be[i])&(sd["frac_toxic_neighbors"]<=be[i+1])
            bd = sd[mask]; n = len(bd)
            if n==0: probs.append(0);counts.append(0);clo.append(0);chi.append(0);continue
            p = bd["is_toxic"].sum()/n; z = 1.96; d = 1+z**2/n
            c = (p+z**2/(2*n))/d; sp = z*np.sqrt((p*(1-p)+z**2/(4*n))/n)/d
            probs.append(float(p));counts.append(n);clo.append(float(max(0,c-sp)));chi.append(float(min(1,c+sp)))
        curves[sub] = {"bin_centers":bc,"probabilities":probs,"counts":counts,"ci_lower":clo,"ci_upper":chi}
    json.dump(curves, open(os.path.join(cfg.RESULTS_DIR,"contagion_curves.json"),"w"), indent=2)

    # Positional
    pos = {}
    for sub in subreddit_names:
        sd = fdf[fdf["subreddit"]==sub]
        if len(sd)<20: continue
        pt = sd[sd["parent_toxic"]]["is_toxic"].mean() if sd["parent_toxic"].any() else 0
        pc = sd[~sd["parent_toxic"]]["is_toxic"].mean() if (~sd["parent_toxic"]).any() else 0
        u1 = sd[sd["num_distinct_toxic_users"]==1]["is_toxic"].mean() if (sd["num_distinct_toxic_users"]==1).any() else 0
        u2 = sd[sd["num_distinct_toxic_users"]>=2]["is_toxic"].mean() if (sd["num_distinct_toxic_users"]>=2).any() else 0
        pos[sub] = {"p_toxic_parent_toxic":float(pt),"p_toxic_parent_clean":float(pc),
                    "p_toxic_1_user":float(u1),"p_toxic_2plus_users":float(u2)}
    json.dump(pos, open(os.path.join(cfg.RESULTS_DIR,"positional_analysis.json"),"w"), indent=2)

    # Stats
    st = {}
    for sub in subreddit_names:
        sd = fdf[fdf["subreddit"]==sub]
        if len(sd)<20: continue
        ct = pd.crosstab(sd["parent_toxic"],sd["is_toxic"])
        c2,pv = (0,1) if ct.shape!=(2,2) else stats.chi2_contingency(ct)[:2]
        X = sd[["frac_toxic_neighbors","num_distinct_toxic_users","thread_depth","score_norm"]].fillna(0).values
        y = sd["is_toxic"].astype(int).values
        try:
            lr = LogisticRegression(max_iter=1000,random_state=42).fit(X,y)
            co = dict(zip(["frac","users","depth","score"],lr.coef_[0].tolist()))
        except: co = {}
        st[sub] = {"chi2":float(c2),"p":float(pv),"lr_coefs":co}
    json.dump(st, open(os.path.join(cfg.RESULTS_DIR,"statistical_tests.json"),"w"), indent=2)

    for sub in subreddit_names[:5]:
        if sub in pos:
            print(f"  r/{sub}: P(tox|par_tox)={pos[sub]['p_toxic_parent_toxic']:.3f} P(tox|par_clean)={pos[sub]['p_toxic_parent_clean']:.3f}")
    return curves, pos, st

# =============================================================================
# STEP 6: AGENTS (same as v2 but with auto-skip)
# =============================================================================

def call_llm(client, model, sp, up):
    for a in range(3):
        try:
            r = client.chat.completions.create(model=model,messages=[{"role":"system","content":sp},{"role":"user","content":up}],temperature=0.3,max_tokens=1000)
            return r.choices[0].message.content
        except Exception as e:
            if a < 2: time.sleep(2**a)
            else: return json.dumps({"error":str(e)})

def parse_json_response(t):
    t = t.strip()
    if t.startswith("```"): t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("```"))
    s,e = t.find("{"),t.rfind("}")+1
    if s>=0 and e>s:
        try: return json.loads(t[s:e])
        except: pass
    return {"error":"parse_failed"}

def format_thread_raw(df, cid, i2r):
    c = df[df["conversation_id"]==cid].sort_values("timestamp")
    lines = []
    for _,r in c.iterrows():
        d,cur,vis = 0,r["reply_to"],set()
        while cur and cur in i2r and cur not in vis: vis.add(cur);d+=1;cur=i2r[cur].get("reply_to")
        lines.append(f"{'  '*d}u/{r['speaker_id']}: {str(r['text'])[:150]}")
    return "\n".join(lines) if lines else "(empty)"

def select_interesting_threads(cfg, subs):
    dfs = [pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{s}.parquet")) for s in subs]
    comb = pd.concat(dfs,ignore_index=True)
    i2r = {r["id"]:r.to_dict() for _,r in comb.iterrows()}
    sl = []
    for cid,cdf in comb.groupby("conversation_id"):
        if len(cdf)<3: continue
        nt = int(cdf["is_toxic"].sum())
        mc = 0
        for _,r in cdf.iterrows():
            cas,ci = 0,r["id"]
            while ci in i2r and i2r[ci].get("is_toxic",False):
                cas+=1;p=i2r[ci].get("reply_to")
                if p==ci or p is None: break
                ci=p
            mc=max(mc,cas)
        first = cdf.sort_values("timestamp").iloc[0]
        sl.append({"conversation_id":cid,"subreddit":cdf["subreddit"].iloc[0],"toxic_rate":nt/len(cdf),
                    "has_toxic_opener":bool(first["is_toxic"]),"rest_toxic_rate":float(cdf.iloc[1:]["is_toxic"].mean()) if len(cdf)>1 else 0,"max_cascade":mc})
    sdf = pd.DataFrame(sl)
    if sdf.empty: return [],comb,i2r
    sel = []; np_ = max(1,cfg.NUM_QUALITATIVE_CASES//4)
    sel.extend(sdf.nlargest(np_,"max_cascade")["conversation_id"].tolist())
    res = sdf[(sdf["has_toxic_opener"])&(sdf["rest_toxic_rate"]<0.2)]
    sel.extend(res.head(np_)["conversation_id"].tolist())
    sel.extend(sdf[(sdf["toxic_rate"]>0.15)&(sdf["toxic_rate"]<0.5)].head(np_)["conversation_id"].tolist())
    sel.extend(sdf[sdf["toxic_rate"]==0].head(np_)["conversation_id"].tolist())
    sel = list(dict.fromkeys(sel))[:cfg.NUM_QUALITATIVE_CASES]
    if len(sel)<cfg.NUM_QUALITATIVE_CASES:
        rem = sdf[~sdf["conversation_id"].isin(sel)]
        sel.extend(rem.sample(min(cfg.NUM_QUALITATIVE_CASES-len(sel),len(rem)),random_state=42)["conversation_id"].tolist())
    return sel,comb,i2r

def step_agents(cfg, subs):
    print("\n"+"="*70+"\nSTEP 6: AGENTS\n"+"="*70)
    if not cfg.OPENAI_API_KEY or cfg.DRY_RUN:
        print(f"Skipping ({'DRY RUN' if cfg.DRY_RUN else 'no key'})."); _gen_dummy(cfg,subs); return
    from openai import OpenAI
    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    sel,comb,i2r = select_interesting_threads(cfg,subs)
    print(f"Selected {len(sel)} threads")
    utox = comb.groupby("speaker_id")["is_toxic"].mean().to_dict()
    ucnt = comb.groupby("speaker_id").size().to_dict()

    # Agent 1
    print("\n>> Agent 1: Norms")
    norms = {}
    for sub in subs:
        df = pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{sub}.parquet"))
        sample = df.sample(min(cfg.NUM_NORM_PROFILE_SAMPLES,len(df)),random_state=42)
        cs = "\n".join(f"{i+1}. {r['text'][:200]}" for i,(_,r) in enumerate(sample.iterrows()))
        try: norms[sub] = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,
            "Sociologist. JSON: {typical_tone,conflict_style,predicted_contagion_type,contagion_justification}",f"r/{sub}:\n{cs}"))
        except: norms[sub] = {"error":"failed"}
    json.dump(norms,open(os.path.join(cfg.RESULTS_DIR,"norm_profiles.json"),"w"),indent=2)

    # Agent 2
    print("\n>> Agent 2: Context")
    ctx = []
    for cid in tqdm(sel,desc="  Context"):
        cc = comb[comb["conversation_id"]==cid]; sub = cc["subreddit"].iloc[0]
        raw = format_thread_raw(cc.sort_values("timestamp").head(cfg.MAX_THREAD_LEN_FOR_LLM),cid,i2r)
        ul = []
        for _,r in cc.iterrows():
            rate = utox.get(r["speaker_id"],0); cnt = ucnt.get(r["speaker_id"],0)
            lbl = "insufficient" if cnt<cfg.MIN_COMMENTS_FOR_USER_PROFILE else ("serial" if rate>0.3 else "context_triggered")
            ul.append(f"  u/{r['speaker_id']}: {lbl} (n={cnt})")
        try:
            p = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,
                "Analyze toxicity. JSON: {serial_vs_triggered:{serial:N,context_triggered:N},dominant_mechanism,trigger_analysis}",
                f"r/{sub}:\n{raw}\n\nProfiles:\n"+"\n".join(ul[:20])))
            p.update({"conversation_id":cid,"subreddit":sub}); ctx.append(p)
        except: ctx.append({"conversation_id":cid,"subreddit":sub,"error":"failed"})
    json.dump(ctx,open(os.path.join(cfg.RESULTS_DIR,"context_analysis.json"),"w"),indent=2)

    # Agent 3
    print("\n>> Agent 3: Deliberation")
    delib = []
    for cid in tqdm(sel,desc="  Delib"):
        cc = comb[comb["conversation_id"]==cid]; sub = cc["subreddit"].iloc[0]
        raw = format_thread_raw(cc.sort_values("timestamp").head(cfg.MAX_THREAD_LEN_FOR_LLM),cid,i2r)
        try:
            sr = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,"Argue SIMPLE CONTAGION. JSON: {argument,evidence:[],confidence:0-1}",f"r/{sub}:\n{raw}"))
            cr = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,"Argue COMPLEX CONTAGION. JSON: {argument,evidence:[],confidence:0-1}",f"r/{sub}:\n{raw}"))
            jr = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,"Judge. JSON: {verdict:simple/complex/ambiguous,reasoning,key_evidence}",
                f"Thread:\n{raw}\n\nSimple:{json.dumps(sr)}\n\nComplex:{json.dumps(cr)}"))
            delib.append({"conversation_id":cid,"subreddit":sub,"judge":jr})
        except: delib.append({"conversation_id":cid,"subreddit":sub,"error":"failed"})
    json.dump(delib,open(os.path.join(cfg.RESULTS_DIR,"deliberation_results.json"),"w"),indent=2)
    v = Counter(r.get("judge",{}).get("verdict","?") for r in delib if "error" not in r)
    print(f"  Verdicts: {dict(v)}")

    # Agent 4
    print("\n>> Agent 4: Counterfactual")
    cfs = []
    ct = [c for c in sel if comb[comb["conversation_id"]==c]["is_toxic"].mean()>0.3][:cfg.NUM_COUNTERFACTUAL_CASES]
    for cid in tqdm(ct,desc="  CF"):
        cc = comb[comb["conversation_id"]==cid].sort_values("timestamp"); sub = cc["subreddit"].iloc[0]
        tc = cc[cc["is_toxic"]]
        if tc.empty: continue
        ft = tc.iloc[0]
        lines = []
        for _,r in cc.head(cfg.MAX_THREAD_LEN_FOR_LLM).iterrows():
            if r["id"]==ft["id"]: lines.append("[REMOVED]")
            else: lines.append(f"u/{r['speaker_id']}: {str(r['text'])[:150]}")
        try:
            resp = parse_json_response(call_llm(client,cfg.OPENAI_MODEL,
                "Moderator removed comment. JSON: {cascade_prevented:bool,confidence:0-1}",
                f"r/{sub}:\n"+"\n".join(lines)+f"\n\nRemoved: \"{str(ft['text'])[:200]}\""))
            resp.update({"conversation_id":cid,"subreddit":sub}); cfs.append(resp)
        except: cfs.append({"conversation_id":cid,"subreddit":sub,"error":"failed"})
    json.dump(cfs,open(os.path.join(cfg.RESULTS_DIR,"counterfactual_results.json"),"w"),indent=2)
    prev = sum(1 for r in cfs if r.get("cascade_prevented") and "error" not in r)
    print(f"  Prevented: {prev}/{len([r for r in cfs if 'error' not in r])}")

def _gen_dummy(cfg,subs):
    norms = {s:{"typical_tone":"casual","conflict_style":"mixed","predicted_contagion_type":"complex"} for s in subs}
    json.dump(norms,open(os.path.join(cfg.RESULTS_DIR,"norm_profiles.json"),"w"),indent=2)
    ctx = [{"subreddit":subs[i%len(subs)],"dominant_mechanism":random.choice(["person-driven","context-driven","mixed"]),
            "serial_vs_triggered":{"serial":1,"context_triggered":2}} for i in range(cfg.NUM_QUALITATIVE_CASES)]
    json.dump(ctx,open(os.path.join(cfg.RESULTS_DIR,"context_analysis.json"),"w"),indent=2)
    delib = [{"subreddit":subs[i%len(subs)],"judge":{"verdict":random.choice(["simple_contagion","complex_contagion","ambiguous"])}} for i in range(cfg.NUM_QUALITATIVE_CASES)]
    json.dump(delib,open(os.path.join(cfg.RESULTS_DIR,"deliberation_results.json"),"w"),indent=2)
    cf = [{"subreddit":subs[i%len(subs)],"cascade_prevented":random.choice([True,False])} for i in range(cfg.NUM_COUNTERFACTUAL_CASES)]
    json.dump(cf,open(os.path.join(cfg.RESULTS_DIR,"counterfactual_results.json"),"w"),indent=2)

# =============================================================================
# STEP 7: FIGURES (18 data-driven, no generic diagrams)
# =============================================================================
def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return default
            return int(float(x))
        return default
    except Exception:
        return default


def _safe_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return default


def _normalize_verdict(v):
    if not isinstance(v, str):
        return "unknown"
    v = v.strip().lower()
    if v in {"simple", "simple_contagion"}:
        return "simple_contagion"
    if v in {"complex", "complex_contagion"}:
        return "complex_contagion"
    if v == "ambiguous":
        return "ambiguous"
    return "unknown"
    
def step_generate_figures(cfg, subreddit_names):
    print("\n"+"="*70+"\nSTEP 7: 18 FIGURES\n"+"="*70)
    sns.set_style("whitegrid")
    def lj(n):
        p=os.path.join(cfg.RESULTS_DIR,n);return json.load(open(p)) if os.path.exists(p) else {}
    def sf(n): plt.savefig(os.path.join(cfg.FIGURES_DIR,n),dpi=cfg.FIGURE_DPI,bbox_inches="tight");plt.close()
    def el(x): return x if isinstance(x,list) else []

    curves=lj("contagion_curves.json"); abl=lj("ablation_results.json")
    pos=lj("positional_analysis.json"); norms=lj("norm_profiles.json")
    ctx=el(lj("context_analysis.json")); delib=el(lj("deliberation_results.json"))
    cf=el(lj("counterfactual_results.json")); tp=lj("test_predictions.json")
    tc = abl.pop("_training_curves",{}) if "_training_curves" in abl else {}
    sc = {s:SUB_CMAP(i/max(1,len(subreddit_names)-1)) for i,s in enumerate(subreddit_names)}

    # Tox rates
    tox_rates = {}
    for sub in subreddit_names:
        df = pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{sub}.parquet"))
        tox_rates[sub] = df["is_toxic"].mean()*100

    # 1: Toxicity ranking
    print("  1/18"); fig,ax=plt.subplots(figsize=(10,max(6,len(subreddit_names)*0.45)))
    ss = sorted(tox_rates.items(),key=lambda x:x[1],reverse=True); names,rates=zip(*ss)
    cols = [plt.cm.RdYlGn_r(r/max(rates)) for r in rates]
    bars=ax.barh(range(len(names)),rates,color=cols)
    ax.set_yticks(range(len(names)));ax.set_yticklabels([f"r/{n}" for n in names],fontsize=9)
    for b,r in zip(bars,rates): ax.text(b.get_width()+0.2,b.get_y()+b.get_height()/2,f"{r:.1f}%",va="center",fontsize=8)
    ax.set_xlabel("% Toxic");ax.set_title("Toxicity Rates Across Subreddits",fontsize=14,fontweight="bold");ax.invert_yaxis();sf("01_toxicity_ranking.png")

    # 2: Tree
    print("  2/18"); fig,ax=plt.subplots(figsize=(12,8))
    try:
        gs=torch.load(os.path.join(cfg.DATA_DIR,f"{subreddit_names[0]}_graphs.pt"),weights_only=False)
        tg=next((g for g in gs if 8<=g.x.shape[0]<=25),gs[0] if gs else None)
        if tg:
            G=nx.DiGraph();n=tg.x.shape[0]
            for i in range(n):G.add_node(i)
            ed=tg.edge_index.numpy()
            for j in range(0,ed.shape[1],2):G.add_edge(ed[0,j],ed[1,j])
            nc=[COLORS["toxic"] if tg.y[i] else COLORS["clean"] for i in range(n)]
            p=nx.spring_layout(G,seed=42,k=2)
            nx.draw(G,p,ax=ax,node_color=nc,node_size=400,with_labels=True,arrows=True,font_size=8,edge_color="gray")
            ax.legend(handles=[mpatches.Patch(facecolor=COLORS["toxic"],label="Toxic"),mpatches.Patch(facecolor=COLORS["clean"],label="Clean")])
    except:ax.text(0.5,0.5,"Error",ha="center")
    ax.set_title("Example Conversation Tree",fontsize=14,fontweight="bold");sf("02_example_tree.png")

    # 3: Distribution (log)
    print("  3/18"); fig,ax=plt.subplots(figsize=(10,6))
    all_s=[];
    for sub in subreddit_names:
        df=pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{sub}.parquet"));all_s.extend(df["toxicity_score"].tolist())
    ax.hist(all_s,bins=100,alpha=0.7,color="#1976D2",edgecolor="white",linewidth=0.5)
    ax.axvline(0.5,color="red",ls="--",lw=2,label="Threshold")
    ax.set_yscale("log");ax.set_xlabel("Toxicity Score");ax.set_ylabel("Count (log)");ax.set_title(f"Toxicity Distribution ({len(all_s):,} comments)",fontsize=14,fontweight="bold");ax.legend();sf("03_distribution.png")

    # 4: Conv stats (top 5)
    print("  4/18"); fig,axes=plt.subplots(1,3,figsize=(16,5))
    top5=[s for s,_ in sorted(tox_rates.items(),key=lambda x:x[1],reverse=True)[:5]]
    for ax,stat,title in zip(axes,["depth","size","toxic"],["Max Depth","Size","Toxic Count"]):
        data,labels=[],[]
        for sub in top5:
            df=pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{sub}.parquet"));vals=[]
            for _,cdf in df.groupby("conversation_id"):
                if stat=="depth":vals.append(cdf["thread_depth"].max() if "thread_depth" in cdf.columns else 0)
                elif stat=="size":vals.append(len(cdf))
                else:vals.append(int(cdf["is_toxic"].sum()))
            data.append(vals);labels.append(f"r/{sub}")
        bp=ax.boxplot(data,labels=labels,patch_artist=True)
        for p in bp["boxes"]:p.set_facecolor("#90CAF9");p.set_alpha(0.7)
        ax.set_title(title,fontweight="bold");ax.tick_params(axis="x",rotation=30,labelsize=8)
    plt.suptitle("Conversation Stats (Top 5 Toxic)",fontweight="bold");plt.tight_layout();sf("04_conversation_stats.png")

    # 5: Training curves
    print("  5/18")
    if tc:
        fig,ax=plt.subplots(figsize=(10,6))
        ac={"full":COLORS["abl_full"],"context_only":COLORS["abl_context"],"all_text":COLORS["abl_text"],"no_graph":COLORS["abl_nograph"],"tabular_only":COLORS["abl_tabular"]}
        for vn,losses in tc.items(): ax.plot(losses,label=vn.replace("_"," ").title(),color=ac.get(vn,"gray"),lw=2)
        ax.set_xlabel("Epoch");ax.set_ylabel("Loss");ax.set_title("Training Curves",fontsize=14,fontweight="bold");ax.legend();sf("05_training_curves.png")

    # 6: Contagion curves (KEY)
    print("  6/18")
    if curves:
        fig,ax=plt.subplots(figsize=(12,8))
        ss2=sorted(tox_rates.items(),key=lambda x:x[1],reverse=True)
        show=[s for s,_ in ss2[:5]]+[s for s,_ in ss2[-2:]];show=[s for s in show if s in curves]
        for sub in show:
            c=curves[sub];color=sc.get(sub)
            ax.plot(c["bin_centers"],c["probabilities"],"o-",color=color,label=f"r/{sub} ({tox_rates.get(sub,0):.0f}%)",lw=2,ms=5)
            ax.fill_between(c["bin_centers"],c["ci_lower"],c["ci_upper"],alpha=0.1,color=color)
        if "combined" in curves:
            c=curves["combined"];ax.plot(c["bin_centers"],c["probabilities"],"s--",color="black",label="Combined",lw=2.5,ms=7)
        ax.set_xlabel("Fraction Toxic Neighbors (Causal)",fontsize=13);ax.set_ylabel("P(Toxic)",fontsize=13)
        ax.set_title("Contagion Threshold Curves",fontsize=15,fontweight="bold");ax.legend(fontsize=9);ax.set_xlim(-0.05,1.05);ax.set_ylim(-0.05,1.05);sf("06_contagion_curves.png")

    # 7: Ablation
    print("  7/18")
    if abl:
        fig,ax=plt.subplots(figsize=(14,6))
        ms=["accuracy","f1_macro","f1_toxic","prec_toxic","rec_toxic","auc","pr_auc"]
        vs=["full","context_only","all_text","no_graph","tabular_only"]
        vc=[COLORS["abl_full"],COLORS["abl_context"],COLORS["abl_text"],COLORS["abl_nograph"],COLORS["abl_tabular"]]
        x=np.arange(len(ms));w=0.15
        for i,(v,c) in enumerate(zip(vs,vc)):
            if v not in abl:continue
            vals=[abl[v].get(m,0) for m in ms];stds=[abl[v].get(f"{m}_std",0) for m in ms]
            bars=ax.bar(x+i*w,vals,w,yerr=stds,label=v.replace("_"," ").title(),color=c,capsize=2)
            for b,val in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{val:.2f}",ha="center",fontsize=5,rotation=45)
        ax.set_xticks(x+w*2);ax.set_xticklabels([m.replace("_","\n") for m in ms],fontsize=8)
        ax.set_title("GAT Ablation (mean+/-std, 3 seeds)",fontsize=14,fontweight="bold");ax.legend(fontsize=7);ax.set_ylim(0,1.2);sf("07_ablation.png")

    # 8: Confusion
    print("  8/18")
    if tp and "confusion_matrix" in tp:
        fig,ax=plt.subplots(figsize=(7,6));cm=np.array(tp["confusion_matrix"])
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=["Clean","Toxic"],yticklabels=["Clean","Toxic"])
        ax.set_xlabel("Predicted");ax.set_ylabel("Actual")
        if cm.shape==(2,2):
            tp_v,fp,fn=cm[1,1],cm[0,1],cm[1,0]
            pr=tp_v/(tp_v+fp) if (tp_v+fp)>0 else 0;re=tp_v/(tp_v+fn) if (tp_v+fn)>0 else 0
            ax.text(0.5,-0.12,f"Toxic Prec: {pr:.3f} | Toxic Rec: {re:.3f}",transform=ax.transAxes,ha="center",fontsize=11,style="italic")
        ax.set_title("Confusion Matrix",fontsize=14,fontweight="bold");sf("08_confusion_matrix.png")

    # 9: ROC + PR
    print("  9/18")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,6))
    vs2=["full","context_only","all_text","no_graph","tabular_only"]
    vc2=[COLORS["abl_full"],COLORS["abl_context"],COLORS["abl_text"],COLORS["abl_nograph"],COLORS["abl_tabular"]]
    for v,c in zip(vs2,vc2):
        if v in abl and "roc_fpr" in abl[v]:
            a1.plot(abl[v]["roc_fpr"],abl[v]["roc_tpr"],color=c,label=f"{v.replace('_',' ').title()} ({abl[v]['auc']:.3f})",lw=2)
    a1.plot([0,1],[0,1],"--",color="gray");a1.set_xlabel("FPR");a1.set_ylabel("TPR");a1.set_title("ROC",fontweight="bold");a1.legend(fontsize=7)
    if tp and "pr_curve" in tp:
        a2.plot(tp["pr_curve"]["recall"],tp["pr_curve"]["precision"],color=COLORS["abl_full"],lw=2,label=f"Full (PR-AUC={abl.get('full',{}).get('pr_auc',0):.3f})")
    a2.set_xlabel("Recall");a2.set_ylabel("Precision");a2.set_title("Precision-Recall",fontweight="bold");a2.legend()
    plt.tight_layout();sf("09_roc_pr.png")

    # 10: Positional
    print("  10/18")
    if pos:
        fig,ax=plt.subplots(figsize=(12,6))
        ps=["p_toxic_parent_toxic","p_toxic_parent_clean","p_toxic_1_user","p_toxic_2plus_users"]
        ls=["Parent\nToxic","Parent\nClean","1 Toxic\nUser","2+ Toxic\nUsers"]
        t5=[s for s,_ in sorted(tox_rates.items(),key=lambda x:x[1],reverse=True)[:5] if s in pos]
        x=np.arange(len(ps));w=0.8/max(1,len(t5))
        for i,sub in enumerate(t5):
            vals=[pos[sub].get(p,0) for p in ps]
            ax.bar(x+i*w,vals,w,label=f"r/{sub}",color=sc.get(sub))
        ax.set_xticks(x+w*(len(t5)-1)/2);ax.set_xticklabels(ls);ax.set_ylabel("P(Toxic)")
        ax.set_title("Positional Analysis",fontsize=14,fontweight="bold");ax.legend(fontsize=8);sf("10_positional.png")

    # 11: User network
    print("  11/18");fig,ax=plt.subplots(figsize=(10,10))
    try:
        ug=pickle.load(open(os.path.join(cfg.DATA_DIR,f"{subreddit_names[0]}_user_network.pkl"),"rb"))
        if ug.number_of_nodes()>50:
            top=sorted(ug.degree(),key=lambda x:x[1],reverse=True)[:50];ug=ug.subgraph([n for n,_ in top])
        p=nx.spring_layout(ug,seed=42,k=0.5);sz=[max(50,d*20) for _,d in ug.degree()]
        tx=[ug.nodes[n].get("avg_toxicity",0) for n in ug.nodes()]
        nx.draw(ug,p,ax=ax,node_size=sz,node_color=tx,cmap=plt.cm.RdYlGn_r,edge_color="lightgray",alpha=0.8,with_labels=False)
        sm=plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,norm=plt.Normalize(0,max(tx) if tx else 1));sm.set_array([]);plt.colorbar(sm,ax=ax,label="Avg Toxicity")
    except:ax.text(0.5,0.5,"Error",ha="center")
    ax.set_title(f"User Network (r/{subreddit_names[0]})",fontsize=14,fontweight="bold");sf("11_user_network.png")

    # 12: Norms
    print("  12/18");fig,ax=plt.subplots(figsize=(14,max(4,len(subreddit_names)*0.35)));ax.axis("off")
    fs=["typical_tone","conflict_style","predicted_contagion_type"]
    show=subreddit_names[:min(10,len(subreddit_names))]
    rows=[[f.replace("_"," ").title()]+[str(norms.get(s,{}).get(f,"N/A")) for s in show] for f in fs]
    cols=["Dimension"]+[f"r/{s}" for s in show]
    t=ax.table(cellText=rows,colLabels=cols,loc="center",cellLoc="center");t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.6)
    for j in range(len(cols)):t[0,j].set_facecolor("#E3F2FD");t[0,j].set_text_props(fontweight="bold")
    ax.set_title("Community Norms (Agent 1)",fontsize=14,fontweight="bold",pad=20);sf("12_norms.png")

    # 13: Serial vs triggered
    print("  13/18");fig,ax=plt.subplots(figsize=(12,6))
    ser, tri = defaultdict(int), defaultdict(int)
    for r in ctx:
        sv = r.get("serial_vs_triggered", {})
        sub = r.get("subreddit", "?")
        ser[sub] += _safe_int(sv.get("serial", 0))
        tri[sub] += _safe_int(sv.get("context_triggered", 0))
    show2=[s for s in subreddit_names if s in ser or s in tri][:10]
    if show2:
        x=np.arange(len(show2))
        ax.bar(x,[ser[s] for s in show2],0.35,label="Serial",color=COLORS["toxic"])
        ax.bar(x+0.35,[tri[s] for s in show2],0.35,label="Context-Triggered",color=COLORS["borderline"])
        ax.set_xticks(x+0.175);ax.set_xticklabels([f"r/{s}" for s in show2],fontsize=7,rotation=30)
    ax.set_ylabel("Count");ax.set_title("Serial vs Context-Triggered (Agent 2)",fontsize=14,fontweight="bold");ax.legend();sf("13_serial_triggered.png")

    # 14: Deliberation
    print("  14/18");fig,ax=plt.subplots(figsize=(8,6))

    vd = Counter(
        _normalize_verdict(r.get("judge", {}).get("verdict", "?"))
        for r in delib if "error" not in r
    )
    vm = {
        "simple_contagion": COLORS["toxic"],
        "complex_contagion": COLORS["abl_context"],
        "ambiguous": COLORS["borderline"],
        "unknown": "gray",
    }
    if vd:
        ax.bar(vd.keys(), vd.values(), color=[vm.get(k, "gray") for k in vd.keys()])

    ax.set_ylabel("Threads");ax.set_title("Deliberation Verdicts (Agent 3)",fontsize=14,fontweight="bold");sf("14_deliberation.png")

    # 15: Counterfactual
    print("  15/18");fig,ax=plt.subplots(figsize=(7,5))
    prev = sum(1 for r in cf if "error" not in r and _safe_bool(r.get("cascade_prevented"), False))
    cont = sum(1 for r in cf if "error" not in r and not _safe_bool(r.get("cascade_prevented"), True))
    ax.bar(["Prevented","Would Continue"],[prev,cont],color=[COLORS["clean"],COLORS["toxic"]])
    ax.set_ylabel("Threads");ax.set_title("Counterfactual Moderation (Agent 4)",fontsize=14,fontweight="bold");sf("15_counterfactual.png")

    # 16-17: Case studies
    print("  16/18");_plot_case(cfg,subreddit_names,"cascade",subreddit_names[0])
    print("  17/18");_plot_case(cfg,subreddit_names,"resilient",subreddit_names[1] if len(subreddit_names)>1 else subreddit_names[0])

    # 18: Summary table
    print("  18/18");fig,ax=plt.subplots(figsize=(16,max(5,len(subreddit_names)*0.3)));ax.axis("off")
    headers=["Subreddit","N","Toxic%","P(t|par_t)","P(t|par_c)","Type","Tone"]
    rows=[]
    for sub in subreddit_names:
        df=pd.read_parquet(os.path.join(cfg.DATA_DIR,f"{sub}.parquet"))
        p=pos.get(sub,{});c=curves.get(sub,{})
        probs=c.get("probabilities",[0]*10)
        early=np.mean(probs[:3]) if len(probs)>=3 else 0;late=np.mean(probs[7:]) if len(probs)>=10 else 0
        ct="Simple" if early>0.3 else "Complex" if late>early*2 else "Mixed"
        rows.append([f"r/{sub}",str(len(df)),f"{df['is_toxic'].mean()*100:.1f}%",
                     f"{p.get('p_toxic_parent_toxic',0):.3f}",f"{p.get('p_toxic_parent_clean',0):.3f}",
                     ct,norms.get(sub,{}).get("typical_tone","N/A")])
    t=ax.table(cellText=rows,colLabels=headers,loc="center",cellLoc="center")
    t.auto_set_font_size(False);t.set_fontsize(7);t.scale(1,1.5)
    for j in range(len(headers)):t[0,j].set_facecolor("#E3F2FD");t[0,j].set_text_props(fontweight="bold")
    ax.set_title("Cross-Subreddit Summary",fontsize=14,fontweight="bold",pad=20);sf("18_summary.png")
    print(f"\n  All 18 figures saved to {cfg.FIGURES_DIR}/")

def _plot_case(cfg,subs,ctype,tsub):
    fig,ax=plt.subplots(figsize=(12,8))
    try:
        gs=torch.load(os.path.join(cfg.DATA_DIR,f"{tsub}_graphs.pt"),weights_only=False);tg=None
        for g in gs:
            tr=g.y.float().mean().item()
            if ctype=="cascade" and tr>0.3 and g.x.shape[0]>=5:tg=g;break
            elif ctype=="resilient" and 0<tr<0.15 and g.x.shape[0]>=5:tg=g;break
        if not tg and gs:tg=gs[0]
        if tg:
            G=nx.DiGraph();n=tg.x.shape[0]
            for i in range(n):G.add_node(i)
            ed=tg.edge_index.numpy()
            for j in range(0,ed.shape[1],2):G.add_edge(ed[0,j],ed[1,j])
            nc=[COLORS["toxic"] if tg.y[i] else COLORS["clean"] for i in range(n)]
            p=nx.spring_layout(G,seed=42,k=2)
            nx.draw(G,p,ax=ax,node_color=nc,node_size=500,with_labels=True,arrows=True,font_size=9,edge_color="gray")
            ax.legend(handles=[mpatches.Patch(facecolor=COLORS["toxic"],label="Toxic"),mpatches.Patch(facecolor=COLORS["clean"],label="Clean")])
    except:ax.text(0.5,0.5,"Error",ha="center")
    label="Toxicity Cascade" if ctype=="cascade" else "Resilient Thread"
    num="16" if ctype=="cascade" else "17"
    ax.set_title(f"Case Study: {label} (r/{tsub})",fontsize=14,fontweight="bold")
    plt.tight_layout();plt.savefig(os.path.join(cfg.FIGURES_DIR,f"{num}_case_{ctype}.png"),dpi=cfg.FIGURE_DPI);plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--api-key",default="")
    parser.add_argument("--skip-agents",action="store_true")
    parser.add_argument("--skip-figures",action="store_true")
    args=parser.parse_args()
    cfg=Config(dry_run=args.dry_run,api_key=args.api_key)
    set_seed(cfg.RANDOM_SEED)
    print("="*70)
    print(f"THE CONTAGION OF CONFLICT v3 | {'DRY' if cfg.DRY_RUN else 'FULL'} | {cfg.DEVICE} | {cfg.NUM_SUBREDDITS} subs | {len(cfg.GAT_SEEDS)} seeds")
    print("="*70)
    total=time.time();timings=[]
    t=time.time();subs=step_download_data(cfg);timings.append(("Load",time.time()-t))
    t=time.time();step_toxicity_scoring(cfg,subs);timings.append(("Toxicity",time.time()-t))
    t=time.time();step_build_graphs(cfg,subs);timings.append(("Graphs",time.time()-t))
    t=time.time();step_train_gat(cfg,subs);timings.append(("GAT",time.time()-t))
    t=time.time();step_contagion_analysis(cfg,subs);timings.append(("Contagion",time.time()-t))
    if not args.skip_agents:
        t=time.time();step_agents(cfg,subs);timings.append(("Agents",time.time()-t))
    else: _gen_dummy(cfg,subs)
    if not args.skip_figures:
        t=time.time();step_generate_figures(cfg,subs);timings.append(("Figures",time.time()-t))
    tot=time.time()-total
    print("\n"+"="*70+"\nDONE\n"+"="*70)
    for n,e in timings:print(f"  {n:<20} {e:>8.1f}s")
    print(f"  {'TOTAL':<20} {tot:>8.1f}s ({tot/60:.1f}m)")

if __name__=="__main__":
    main()
