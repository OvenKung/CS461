"""
Fine-tune Sentence Transformer เพื่อทำความเข้าใจภาพยนตร์ดีขึ้น
ใช้ Multiple Negatives Ranking Loss เพื่อเรียนรู้ความคล้ายคลึง

GPU จะช่วยให้เทรนเร็วขึ้นมาก (10-20 เท่า)
Mac M2 ใช้เวลาประมาณ 1-1.5 ชั่วโมง (GPU) หรือ 3-4 ชั่วโมง (CPU)
"""

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import os

# ตั้ง random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def get_best_device():
    """เลือก device ที่ดีที่สุด - GPU ก่อนเสมอ"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ ใช้ CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("✅ ใช้ Apple Silicon GPU (MPS)")
        print("   การเทรนจะเร็วขึ้นกว่า CPU (ใช้ batch_size=4 เพื่อประหยัด memory)")
    else:
        device = 'cpu'
        print("⚠️  ใช้ CPU - การเทรนจะใช้เวลานาน")
        print("💡 แนะนำ: ใช้เครื่องที่มี GPU หรือ Apple Silicon")
    return device


def create_training_examples(movies_df, num_examples=5000):
    """
    สร้างตัวอย่างการเทรนจากข้อมูลหนังจริง
    
    กลยุทธ์:
    - Positive pairs: หนังแนวเดียวกัน, ผู้กำกับเดียวกัน, keywords คล้ายกัน
    - รูปแบบ (query, positive) pairs สำหรับ Multiple Negatives Ranking Loss
    """
    print("\n📝 กำลังสร้างตัวอย่างการเทรนจากข้อมูลหนังจริง...")
    
    examples = []
    
    # 1. Genre-based pairs (หนังแนวเดียวกันควรใกล้กัน)
    print("   🎬 สร้าง genre-based pairs...")
    for _ in tqdm(range(num_examples // 3), desc="Genre"):
        # สุ่มแนวหนัง
        all_genres = set()
        for genres in movies_df['genres']:
            all_genres.update(genres)
        
        if len(all_genres) == 0:
            continue
            
        genre = random.choice(list(all_genres))
        
        # หาหนังในแนวนี้
        genre_movies = movies_df[movies_df['genres'].apply(lambda x: genre in x)]
        if len(genre_movies) >= 2:
            sample = genre_movies.sample(2)
            query = f"{genre} movie with great story"
            positive = f"{sample.iloc[0]['title']}. {sample.iloc[0]['overview']}"
            examples.append(InputExample(texts=[query, positive]))
    
    # 2. Director-based pairs (หนังผู้กำกับเดียวกันมีสไตล์คล้าย)
    print("   🎥 สร้าง director-based pairs...")
    for _ in tqdm(range(num_examples // 3), desc="Director"):
        directors = movies_df['director'].value_counts()
        # เลือกผู้กำกับที่มีหนังมากกว่า 2 เรื่อง
        prolific_directors = directors[directors >= 2].index.tolist()
        if prolific_directors:
            director = random.choice(prolific_directors)
            director_movies = movies_df[movies_df['director'] == director]
            if len(director_movies) >= 2:
                sample = director_movies.sample(2)
                query = f"movie directed by {director}"
                positive = f"{sample.iloc[0]['title']}. {sample.iloc[0]['overview']}"
                examples.append(InputExample(texts=[query, positive]))
    
    # 3. Keyword-based pairs (keywords คล้าย = เนื้อหาคล้าย)
    print("   🔑 สร้าง keyword-based pairs...")
    for _ in tqdm(range(num_examples // 3), desc="Keyword"):
        movie1 = movies_df.sample(1).iloc[0]
        if isinstance(movie1['keywords'], list) and len(movie1['keywords']) > 0:
            # หาหนังที่มี keywords ตรงกัน
            keyword = random.choice(movie1['keywords'])
            matching = movies_df[movies_df['keywords'].apply(
                lambda x: isinstance(x, list) and keyword in x
            )]
            if len(matching) >= 2:
                movie2 = matching[matching['title'] != movie1['title']]
                if len(movie2) > 0:
                    movie2 = movie2.sample(1).iloc[0]
                    query = f"movie about {keyword}"
                    positive = f"{movie2['title']}. {movie2['overview']}"
                    examples.append(InputExample(texts=[query, positive]))
    
    print(f"\n✅ สร้างตัวอย่างการเทรน {len(examples):,} ตัวอย่าง")
    return examples


def finetune_model(
    model_name='BAAI/bge-base-en-v1.5',
    epochs=3,
    batch_size=32,         # เพิ่มเป็น 32 สำหรับ 32GB RAM
    warmup_steps=500,
    output_path='data/finetuned_model'
):
    """
    Fine-tune โมเดลด้วยข้อมูลหนังจริง
    
    พารามิเตอร์:
        epochs: จำนวนรอบการเทรน (3-4 รอบเพียงพอ)
        batch_size: ขนาด batch (16 สำหรับ GPU ปานกลาง, 32+ สำหรับ GPU แรง)
        warmup_steps: Learning rate warmup (500-1000 ดี)
    """
    print("=" * 80)
    print("🎓 Fine-tuning Sentence Transformer สำหรับหนัง")
    print("=" * 80)
    
    # โหลดข้อมูล
    print("\n📥 โหลดข้อมูลหนัง...")
    movies_df = pd.read_pickle('data/movies.pkl')
    print(f"✅ โหลดหนัง {len(movies_df):,} เรื่อง")
    
    # เตรียม device
    device = get_best_device()
    
    # โหลด base model
    print(f"\n🧠 โหลด base model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # สร้างตัวอย่างการเทรน
    train_examples = create_training_examples(movies_df, num_examples=5000)
    
    # ปรับ batch_size ตาม device
    if device == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:
            batch_size = 4
            print(f"⚠️  ปรับ batch_size เป็น {batch_size} เพื่อ VRAM")
    # MPS: ใช้ batch size ที่กำหนด (สำหรับ 32GB RAM ให้ใช้ batch_size=32)
    # else:
    #     # CPU และ MPS ใช้ batch size เล็ก
    #     batch_size = min(batch_size, 4)
    #     print(f"⚠️  ปรับ batch_size เป็น {batch_size} สำหรับ {device.upper()}")
    #     if device == 'cpu':
    #         print("   (ลด batch size เพื่อให้เทรนได้เร็วขึ้น)")
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # กำหนด loss function
    # Multiple Negatives Ranking Loss: เรียนรู้ว่าหนังเรื่องไหนใกล้เคียงกัน
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # คำนวณ warmup steps
    total_steps = len(train_dataloader) * epochs
    
    print(f"\n🎯 การตั้งค่าการเทรน:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training examples: {len(train_examples):,}")
    print(f"   Steps per epoch: {len(train_dataloader)}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Device: {device.upper()}")
    
    # ประมาณเวลา
    if device == 'cuda':
        time_estimate = total_steps * 0.5 / 60
        print(f"\n⚡ GPU Training - คาดว่าจะใช้เวลา ~{time_estimate:.1f} นาที")
    elif device == 'mps':
        time_estimate = total_steps * 3.0 / 60
        print(f"\n⚡ MPS Training - คาดว่าจะใช้เวลา ~{time_estimate:.1f} นาที")
        print("   (batch_size=4 จะช้ากว่าปกติแต่ไม่หมด memory)")
    else:
        time_estimate = total_steps * 4.5 / 60
        print(f"\n🐢 CPU Training - คาดว่าจะใช้เวลา ~{time_estimate:.1f} นาที")
        print("   (Mac M2 ควรเสร็จภายใน 1-2 ชั่วโมง)")
    
    # สร้าง output directory
    os.makedirs(output_path, exist_ok=True)
    
    # เทรนโมเดล
    print("\n🚀 เริ่มการเทรน...\n")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=True,
    )
    
    print(f"\n✅ Fine-tuning เสร็จสมบูรณ์!")
    print(f"💾 โมเดลที่ปรับแต่งแล้วบันทึกที่: {output_path}")
    
    print("\n📝 ขั้นตอนต่อไป:")
    print("   1. รัน: python generate_embeddings.py (จะใช้โมเดลใหม่โดยอัตโนมัติ)")
    print("   2. รัน: python app.py (จะโหลดโมเดลที่ fine-tune แล้วโดยอัตโนมัติ)")
    print("   3. ทดสอบความแม่นยำที่ดีขึ้น!")
    
    return model


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🎬 Movie Recommendation Model Fine-tuning")
    print("=" * 80)
    
    # Fine-tune โมเดล (ปรับสำหรับ 32GB RAM)
    finetuned_model = finetune_model(
        model_name='BAAI/bge-base-en-v1.5',  # ใช้ SOTA model
        epochs=3,              # 3 epochs เพียงพอสำหรับ fine-tuning
        batch_size=32,         # เพิ่มเป็น 32 สำหรับ 32GB RAM (เร็วกว่ามาก!)
        warmup_steps=500,
        output_path='data/finetuned_model'
    )
    
    print("\n" + "=" * 80)
    print("🎉 โมเดลพร้อมใช้งาน!")
    print("=" * 80)
