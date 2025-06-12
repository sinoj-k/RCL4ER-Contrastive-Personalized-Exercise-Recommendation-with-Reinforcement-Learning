# RCL4ER Full Implementation with ASSIST09 Dataset and Top-K Recommendation by Student ID

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Configs and Hyperparameters
# -----------------------------
EMBED_DIM = 64
SEQ_LEN = 50
TOP_K = 5

# -----------------------------
# Dataset Loader for ASSIST09
# -----------------------------
def load_assist09(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['skill_id'])
    label_enc_skill = LabelEncoder()
    label_enc_user = LabelEncoder()
    df['skill_id'] = label_enc_skill.fit_transform(df['skill_id'])
    df['user_id'] = label_enc_user.fit_transform(df['user_id'])

    if 'start_time' in df.columns:
        df = df.sort_values(by=['user_id', 'start_time'])
    else:
        df = df.sort_values(by=['user_id'])
        print("Warning: 'start_time' column not found. Sorting only by 'user_id'.")

    student_seqs = {}
    for row in df.itertuples():
        user, skill, correct = row.user_id, row.skill_id, row.correct
        if user not in student_seqs:
            student_seqs[user] = []
        student_seqs[user].append((skill, correct))
    return list(student_seqs.values()), df['skill_id'].nunique(), student_seqs, label_enc_user

# -----------------------------
# Dataset Class
# -----------------------------
class RCL4ERDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_seq_len=SEQ_LEN):
        self.data = data
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx][-self.max_seq_len:]
        q_ids = [q for q, _ in seq]
        rs = [r for _, r in seq]
        q_ids = q_ids + [0] * (self.max_seq_len - len(q_ids))
        rs = rs + [0] * (self.max_seq_len - len(rs))
        return torch.tensor(q_ids), torch.tensor(rs)

# -----------------------------
# Data Augmentation
# -----------------------------
def augment(sequence, mode='mask', ratio=0.2):
    aug_seq = sequence.copy()
    L = len(aug_seq)
    idxs = random.sample(range(L), max(1, int(L * ratio)))
    if mode == 'mask':
        for idx in idxs:
            aug_seq[idx] = (0, 0)
    elif mode == 'permute':
        sub = [aug_seq[i] for i in idxs]
        random.shuffle(sub)
        for i, idx in enumerate(idxs):
            aug_seq[idx] = sub[i]
    elif mode == 'replace':
        for idx in idxs:
            q, r = aug_seq[idx]
            new_q = (q + random.randint(1, 5)) % VOCAB_SIZE
            aug_seq[idx] = (new_q, r)
    return aug_seq

# -----------------------------
# Embedding + Encoders
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.q_embed = nn.Embedding(vocab_size, embed_dim)
        self.qa_embed = nn.Embedding(2 * vocab_size, embed_dim)
        self.lstm_q = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.lstm_qa = nn.LSTM(embed_dim, embed_dim, batch_first=True)

    def forward(self, q_seq, qa_seq):
        q_emb = self.q_embed(q_seq)
        qa_emb = self.qa_embed(qa_seq)
        q_enc, _ = self.lstm_q(q_emb)
        qa_enc, _ = self.lstm_qa(qa_emb)
        return q_enc, qa_enc

# -----------------------------
# Contrastive Learning Loss
# -----------------------------
def contrastive_loss(z1, z2, temp=0.5):
    sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2)
    logits = sim / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)

# -----------------------------
# Q-Learning Head
# -----------------------------
class QNet(nn.Module):
    def __init__(self, embed_dim, action_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.fc2 = nn.Linear(embed_dim // 2, action_size)

    def forward(self, state):
        out = torch.relu(self.fc1(state))
        return self.fc2(out)

# -----------------------------
# Main Model
# -----------------------------
class RCL4ER(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim)
        self.q_net = QNet(embed_dim * 2, vocab_size)
        self.out = nn.Linear(embed_dim * 2, vocab_size)

    def forward(self, q_seq, r_seq):
        qa_seq = q_seq + r_seq.type(torch.LongTensor) * VOCAB_SIZE
        h_q, h_qa = self.encoder(q_seq, qa_seq)
        state = torch.cat([h_q[:, -1], h_qa[:, -1]], dim=1)
        logits = self.out(state)
        q_values = self.q_net(state)
        return logits, q_values, state

# -----------------------------
# Training Step
# -----------------------------
def train_step(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for q_seq, r_seq in data_loader:
        optimizer.zero_grad()
        logits, q_vals, state = model(q_seq, r_seq)
        labels = q_seq[:, -1]
        loss_s = F.cross_entropy(logits, labels)

        aug_seq1 = [augment(list(zip(q.tolist(), r.tolist())), 'mask') for q, r in zip(q_seq, r_seq)]
        aug_seq2 = [augment(list(zip(q.tolist(), r.tolist())), 'permute') for q, r in zip(q_seq, r_seq)]
        aug_q1 = torch.stack([torch.tensor([x[0] for x in seq]) for seq in aug_seq1])
        aug_r1 = torch.stack([torch.tensor([x[1] for x in seq]) for seq in aug_seq1])
        aug_q2 = torch.stack([torch.tensor([x[0] for x in seq]) for seq in aug_seq2])
        aug_r2 = torch.stack([torch.tensor([x[1] for x in seq]) for seq in aug_seq2])

        _, _, z1 = model(aug_q1, aug_r1)
        _, _, z2 = model(aug_q2, aug_r2)
        loss_cl = contrastive_loss(z1, z2)

        q_target = q_vals.detach().clone()
        q_target = q_target.gather(1, labels.unsqueeze(1)).squeeze()
        loss_q = F.mse_loss(q_vals.gather(1, labels.unsqueeze(1)).squeeze(), q_target)

        loss = loss_s + loss_cl + loss_q
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# -----------------------------
# Recommend Top-K Exercises
# -----------------------------
def recommend_top_k(model, q_seq, r_seq, k=TOP_K):
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(q_seq, r_seq)
        topk = torch.topk(logits, k=k, dim=1)
        return topk.indices

# -----------------------------
# Recommend for Given Student ID
# -----------------------------
def recommend_for_student_id(model, student_id, student_seqs, label_enc_user, k=TOP_K):
    if student_id not in label_enc_user.classes_:
        raise ValueError(f"Student ID '{student_id}' not found in dataset.")
    encoded_id = label_enc_user.transform([student_id])[0]
    student_seq = student_seqs[encoded_id][-SEQ_LEN:]
    q_ids = [q for q, _ in student_seq]
    r_ids = [r for _, r in student_seq]
    q_ids += [0] * (SEQ_LEN - len(q_ids))
    r_ids += [0] * (SEQ_LEN - len(r_ids))
    q_tensor = torch.tensor(q_ids).unsqueeze(0)
    r_tensor = torch.tensor(r_ids).unsqueeze(0)
    topk = recommend_top_k(model, q_tensor, r_tensor, k)
    return topk.tolist()[0]

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    path = "/content/drive/MyDrive/dataset/assist09_cleaned.csv"
    data, VOCAB_SIZE, student_seq_map, user_encoder = load_assist09(path)
    dataset = RCL4ERDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = RCL4ER(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        loss = train_step(model, loader, optimizer)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    student_id_input = 78068  # Replace with a real student ID from the CSV
    try:
        recommendations = recommend_for_student_id(model, student_id_input, student_seq_map, user_encoder)
        print(f"Top-{TOP_K} Recommendations for student '{student_id_input}':", recommendations)
    except ValueError as e:
        print(e)
