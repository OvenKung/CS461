"""
Fine-tune Sentence Transformer สำหรับ Movie Recommendation System
ใช้ Contrastive Learning และ Triplet Loss

การทำงาน:
1. สร้าง training pairs จากข้อมูลหนัง (positive/negative pairs)
2. Fine-tune all-mpnet-base-v2 model
3. Evaluate ด้วย similarity metrics
4. Save fine-tuned model และ embeddings
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import os
from datetime import datetime
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ตั้งค่า random seed สำหรับ reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 80)
print("🎓 Fine-tuning Movie Recommendation Model")
print("=" * 80)

# ========================
# 1. โหลดข้อมูล
# ========================
print("\n📊 กำลังโหลดข้อมูล...")
movies_df = pd.read_pickle('data/movies.pkl')
print(f"✅ โหลดสำเร็จ: {len(movies_df):,} เรื่อง")

# แบ่งข้อมูลเป็น train/validation/test
train_df, temp_df = train_test_split(movies_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"📦 Train: {len(train_df):,} เรื่อง")
print(f"📦 Validation: {len(val_df):,} เรื่อง")
print(f"📦 Test: {len(test_df):,} เรื่อง")


# ========================
# 2. สร้าง Training Pairs
# ========================
def create_positive_pair(df, row):
    """
    สร้าง Positive Pair (หนังที่คล้ายกัน) โดยพิจารณา:
    - แนวหนังเดียวกัน (genres overlap)
    - ผู้กำกับคนเดียวกัน
    - นักแสดงคนเดียวกัน
    - คีย์เวิร์ดเดียวกัน
    """
    candidate_scores = []
    
    for idx, other_row in df.iterrows():
        if idx == row.name:
            continue
        
        score = 0.0
        
        # Genre overlap (สำคัญที่สุด)
        genre_overlap = len(set(row['genres']) & set(other_row['genres']))
        score += genre_overlap * 0.4
        
        # ผู้กำกับคนเดียวกัน
        if row['director'] and other_row['director'] and row['director'] == other_row['director']:
            score += 0.3
        
        # นักแสดงคนเดียวกัน
        cast_overlap = len(set(row['cast'][:5]) & set(other_row['cast'][:5]))
        score += cast_overlap * 0.1
        
        # คีย์เวิร์ดเดียวกัน
        keyword_overlap = len(set(row['keywords'][:10]) & set(other_row['keywords'][:10]))
        score += keyword_overlap * 0.05
        
        if score > 0.3:  # มีความคล้ายกันพอสมควร
            candidate_scores.append((idx, score))
    
    if candidate_scores:
        # เลือกหนังที่คล้ายที่สุด
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        selected_idx = candidate_scores[0][0]
        return df.loc[selected_idx], candidate_scores[0][1]
    
    return None, 0.0


def create_negative_pair(df, row):
    """
    สร้าง Negative Pair (หนังที่ไม่คล้ายกัน):
    - แนวหนังต่างกันโดยสิ้นเชิง
    - ไม่มีนักแสดง/ผู้กำกับคนเดียวกัน
    """
    candidates = df[~df['genres'].apply(
        lambda x: bool(set(x) & set(row['genres']))
    )]
    
    if len(candidates) > 0:
        return candidates.sample(1).iloc[0]
    
    return None


def create_training_examples(df, num_pairs=10000, pair_type='both'):
    """
    สร้าง InputExample สำหรับ training
    
    Args:
        df: DataFrame ของหนัง
        num_pairs: จำนวน pairs ที่ต้องการ
        pair_type: 'positive', 'negative', 'both'
    """
    examples = []
    
    print(f"\n🔧 กำลังสร้าง {num_pairs} training pairs...")
    
    sampled_movies = df.sample(min(num_pairs, len(df)), random_state=42)
    
    for idx, (_, row) in enumerate(sampled_movies.iterrows()):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{num_pairs}")
        
        # Positive pair
        if pair_type in ['positive', 'both']:
            pos_movie, similarity = create_positive_pair(df, row)
            if pos_movie is not None:
                examples.append(InputExample(
                    texts=[row['rich_description'], pos_movie['rich_description']],
                    label=min(similarity, 1.0)  # ค่าความคล้าย 0-1
                ))
        
        # Negative pair
        if pair_type in ['negative', 'both']:
            neg_movie = create_negative_pair(df, row)
            if neg_movie is not None:
                examples.append(InputExample(
                    texts=[row['rich_description'], neg_movie['rich_description']],
                    label=0.0  # ไม่คล้ายกันเลย
                ))
    
    print(f"✅ สร้างเสร็จ: {len(examples)} pairs")
    return examples


print("\n🎯 สร้าง Training Examples...")
train_examples = create_training_examples(train_df, num_pairs=8000, pair_type='both')

print("\n🎯 สร้าง Validation Examples...")
val_examples = create_training_examples(val_df, num_pairs=1500, pair_type='both')

print("\n🎯 สร้าง Test Examples...")
test_examples = create_training_examples(test_df, num_pairs=1000, pair_type='both')


# ========================
# 3. เตรียม DataLoader
# ========================
print("\n📦 เตรียม DataLoader...")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
print(f"  Train batches: {len(train_dataloader)}")


# ========================
# 4. โหลด Base Model
# ========================
print("\n🧠 กำลังโหลด base model...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"📱 ใช้ device: {device}")
model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
print("✅ โหลด BAAI/bge-base-en-v1.5 สำเร็จ (SOTA Model)")

# ล้าง cache เพื่อประหยัด RAM
if device == "mps":
    torch.mps.empty_cache()


# ========================
# 5. สร้าง Evaluator
# ========================
print("\n🧪 สร้าง Evaluators...")

# ใช้แค่ Embedding Similarity Evaluator เพื่อประหยัด RAM
val_sentences1 = [ex.texts[0] for ex in val_examples[:600]]
val_sentences2 = [ex.texts[1] for ex in val_examples[:600]]
val_scores = [ex.label for ex in val_examples[:600]]

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    val_sentences1,
    val_sentences2,
    val_scores,
    name='movie-validation'
)

print("✅ ใช้ EmbeddingSimilarityEvaluator (ประหยัด RAM)")


# ========================
# 6. กำหนด Loss Function
# ========================
print("\n⚙️  กำหนด Loss Function...")
train_loss = losses.CosineSimilarityLoss(model)
print("✅ ใช้ CosineSimilarityLoss")


# ========================
# 7. Fine-tuning!
# ========================
output_path = 'models/movie-mpnet-finetuned'
os.makedirs(output_path, exist_ok=True)

print("\n" + "=" * 80)
print("🔥 เริ่ม Fine-tuning...")
print("=" * 80)

training_config = {
    'base_model': 'all-mpnet-base-v2',
    'train_examples': len(train_examples),
    'val_examples': len(val_examples),
    'batch_size': 32,
    'epochs': 4,
    'warmup_steps': 50,
    'evaluation_steps': 400,
    'save_best_model': True,
    'device': device,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# บันทึก config
with open(f'{output_path}/training_config.json', 'w', encoding='utf-8') as f:
    json.dump(training_config, f, indent=2, ensure_ascii=False)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=50,
    evaluator=evaluator,
    evaluation_steps=400,
    output_path=output_path,
    save_best_model=True,
    show_progress_bar=True
)

print("\n✅ Fine-tuning เสร็จสมบูรณ์!")

# ล้าง cache หลัง training
if device == "mps":
    torch.mps.empty_cache()


# ========================
# 8. Evaluation บน Test Set
# ========================
print("\n" + "=" * 80)
print("📊 Evaluating บน Test Set...")
print("=" * 80)

# โหลด best model
best_model = SentenceTransformer(output_path)

# Test Embedding Similarity
test_sentences1 = [ex.texts[0] for ex in test_examples[:1000]]
test_sentences2 = [ex.texts[1] for ex in test_examples[:1000]]
test_scores = [ex.label for ex in test_examples[:1000]]

test_evaluator = evaluation.EmbeddingSimilarityEvaluator(
    test_sentences1,
    test_sentences2,
    test_scores,
    name='movie-test'
)

print("\n🧪 Test Set Results:")
test_result = test_evaluator(best_model, output_path=output_path)

# ========================
# 9. คำนวณ Pairwise Metrics และสร้างกราฟ
# ========================
print("\n" + "=" * 80)
print("📊 คำนวณ Pairwise Metrics และสร้างกราฟ...")
print("=" * 80)

def compute_pair_metrics(model, examples, threshold=0.5, max_samples=2000):
    """
    คำนวณ Precision, Recall, F1, Accuracy, Spearman
    """
    subset = examples[:max_samples]
    t1 = [e.texts[0] for e in subset]
    t2 = [e.texts[1] for e in subset]
    
    print(f"  Encoding {len(t1)} pairs...")
    emb1 = model.encode(t1, batch_size=16, normalize_embeddings=True, show_progress_bar=True, device=device)
    emb2 = model.encode(t2, batch_size=16, normalize_embeddings=True, show_progress_bar=False, device=device)
    
    sims = [float(np.dot(a, b)) for a, b in zip(emb1, emb2)]
    labels = [float(e.label) for e in subset]
    
    # แปลงเป็น binary classification (threshold สำหรับ label และ similarity)
    true_bin = [1 if l >= 0.3 else 0 for l in labels]
    pred_bin = [1 if s >= threshold else 0 for s in sims]
    
    tp = sum(1 for t, p in zip(true_bin, pred_bin) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_bin, pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_bin, pred_bin) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_bin, pred_bin) if t == 0 and p == 0)
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    
    spearman_corr, _ = spearmanr(labels, sims)
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "spearman": float(spearman_corr),
        "sims": sims,
        "labels": labels,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }

print("\n📈 Fine-tuned Model Metrics:")
pair_metrics = compute_pair_metrics(best_model, test_examples, threshold=0.5, max_samples=2000)

# ล้าง cache
if device == "mps":
    torch.mps.empty_cache()

# สร้างโฟลเดอร์สำหรับกราฟ
metrics_dir = f"{output_path}/metrics"
os.makedirs(metrics_dir, exist_ok=True)

# กราฟ 1: Similarity Distribution
print("\n📊 สร้างกราฟ Similarity Distribution...")
plt.figure(figsize=(10, 6))
sns.histplot(pair_metrics["sims"], bins=50, kde=True, color="#2563eb")
plt.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold 0.5")
plt.title("Cosine Similarity Distribution (Fine-tuned Model)", fontsize=14, fontweight='bold')
plt.xlabel("Cosine Similarity", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
dist_path = f"{metrics_dir}/similarity_distribution.png"
plt.tight_layout()
plt.savefig(dist_path, dpi=150)
plt.close()
print(f"  ✅ บันทึกที่: {dist_path}")

# กราฟ 2: Label vs Similarity Scatter
print("\n📊 สร้างกราฟ Label vs Similarity...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pair_metrics["labels"], y=pair_metrics["sims"], s=20, alpha=0.4, color="#10b981")
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label="Perfect Correlation")
plt.title("Ground Truth Label vs Predicted Similarity", fontsize=14, fontweight='bold')
plt.xlabel("Ground Truth Label (0-1)", fontsize=12)
plt.ylabel("Cosine Similarity", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
scatter_path = f"{metrics_dir}/label_vs_similarity.png"
plt.tight_layout()
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"  ✅ บันทึกที่: {scatter_path}")

# กราฟ 3: Confusion Matrix
print("\n📊 สร้างกราฟ Confusion Matrix...")
cm = pair_metrics["confusion_matrix"]
cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix (Threshold=0.5)", fontsize=14, fontweight='bold')
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
cm_path = f"{metrics_dir}/confusion_matrix.png"
plt.tight_layout()
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"  ✅ บันทึกที่: {cm_path}")

print("\n📈 Pairwise Classification Metrics:")
print(f"  Threshold:  {pair_metrics['threshold']}")
print(f"  Precision:  {pair_metrics['precision']:.4f}")
print(f"  Recall:     {pair_metrics['recall']:.4f}")
print(f"  F1 Score:   {pair_metrics['f1']:.4f}")
print(f"  Accuracy:   {pair_metrics['accuracy']:.4f}")
print(f"  Spearman:   {pair_metrics['spearman']:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TP: {cm['tp']}, FP: {cm['fp']}")
print(f"    FN: {cm['fn']}, TN: {cm['tn']}")


# ========================
# 10. สร้าง Embeddings ใหม่
# ========================
print("\n" + "=" * 80)
print("🧮 กำลังสร้าง embeddings ใหม่สำหรับข้อมูลทั้งหมด...")
print("=" * 80)

all_descriptions = movies_df['rich_description'].tolist()
new_embeddings = best_model.encode(
    all_descriptions,
    show_progress_bar=True,
    batch_size=32,
    normalize_embeddings=True
)

# บันทึก embeddings
embedding_path = 'data/movie_embeddings_finetuned.npy'
np.save(embedding_path, new_embeddings)
print(f"✅ บันทึก embeddings ที่: {embedding_path}")
print(f"   Shape: {new_embeddings.shape}")


# ========================
# 10. เปรียบเทียบ Original vs Fine-tuned
# ========================
print("\n" + "=" * 80)
print("📈 เปรียบเทียบ Performance: Original vs Fine-tuned")
print("=" * 80)

# โหลด original model
original_model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# ทดสอบกับ test set
print("\n🔵 Original Model (BAAI/bge-base-en-v1.5 - Base):")
original_result = test_evaluator(original_model)
original_score = original_result if isinstance(original_result, float) else original_result.get('spearman_cosine', 0.0)

print("\n🟢 Fine-tuned Model:")
finetuned_result = test_evaluator(best_model)
finetuned_score = finetuned_result if isinstance(finetuned_result, float) else finetuned_result.get('spearman_cosine', 0.0)

# คำนวณการปรับปรุง
improvement = {
    'spearman_correlation': finetuned_score - original_score
}

print("\n📊 Summary:")
print(f"  Original Spearman: {original_score:.4f}")
print(f"  Fine-tuned Spearman: {finetuned_score:.4f}")
print(f"  Improvement: {improvement['spearman_correlation']:.4f} ({improvement['spearman_correlation']*100:.2f}%)")


# ========================
# 11. บันทึก Evaluation Report
# ========================
evaluation_report = {
    'training_config': training_config,
    'test_results': {
        'original_model': float(original_score),
        'finetuned_model': float(finetuned_score),
        'improvement': float(improvement['spearman_correlation'])
    },
    'pairwise_metrics': {
        'threshold': pair_metrics['threshold'],
        'precision': pair_metrics['precision'],
        'recall': pair_metrics['recall'],
        'f1': pair_metrics['f1'],
        'accuracy': pair_metrics['accuracy'],
        'spearman': pair_metrics['spearman'],
        'confusion_matrix': pair_metrics['confusion_matrix'],
        'plots': {
            'similarity_distribution': dist_path,
            'label_vs_similarity': scatter_path,
            'confusion_matrix': cm_path
        }
    },
    'model_info': {
        'base_model': 'BAAI/bge-base-en-v1.5',
        'finetuned_path': output_path,
        'embedding_path': embedding_path,
        'embedding_dim': 768,
        'num_movies': len(movies_df)
    },
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

report_path = f'{output_path}/evaluation_report.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

print(f"\n💾 บันทึก evaluation report ที่: {report_path}")


# ========================
# 12. สรุปและคำแนะนำ
# ========================
print("\n" + "=" * 80)
print("✅ การ Fine-tuning เสร็จสมบูรณ์!")
print("=" * 80)

print("\n📁 ไฟล์ที่สร้างขึ้น:")
print(f"  1. Model: {output_path}/")
print(f"  2. Embeddings: {embedding_path}")
print(f"  3. Config: {output_path}/training_config.json")
print(f"  4. Report: {report_path}")
print(f"  5. Plots: {metrics_dir}/")
print(f"     - similarity_distribution.png")
print(f"     - label_vs_similarity.png")
print(f"     - confusion_matrix.png")

print("\n🔧 วิธีใช้งาน Fine-tuned Model ใน app.py:")
print("  แก้บรรทัด 317-318:")
print("  ")
print("  # เดิม:")
print("  model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')")
print("  movie_vectors = np.load('data/movie_embeddings.npy')")
print("  ")
print("  # ใหม่:")
print("  model = SentenceTransformer('models/movie-mpnet-finetuned', device='cpu')")
print("  movie_vectors = np.load('data/movie_embeddings_finetuned.npy')")

print("\n🚀 จากนั้นรัน Flask app ตามปกติ:")
print("  python app.py")

print("\n" + "=" * 80)
