"""
การสร้าง Embeddings ด้วย Models ที่ทันสมัยที่สุด
สำหรับวิชา Neural Network & Deep Learning

Models ที่ใช้:
1. all-mpnet-base-v2 - คุณภาพดีที่สุด (768 มิติ, ช้ากว่า)
2. paraphrase-multilingual-mpnet-base-v2 - รองรับหลายภาษา (768 มิติ)
3. all-MiniLM-L6-v2 - เร็วพื้นฐาน (384 มิติ)

ฟีเจอร์:
- ประมวลผลแบบ batch เพื่อประสิทธิภาพ
- เร่งด้วย GPU (ถ้ามี)
- ติดตามความคืบหน้า
- ปรับแต่งการใช้ memory
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

def generate_advanced_embeddings(
    model_name='BAAI/bge-base-en-v1.5',
    batch_size=32,
    use_gpu=False
):
    """
    สร้าง embeddings คุณภาพสูงด้วย models ที่ทันสมัยที่สุด
    
    พารามิเตอร์:
        model_name: Model ที่จะใช้
            - 'all-mpnet-base-v2': คุณภาพดีที่สุด (768 มิติ) ⭐ แนะนำ
            - 'paraphrase-multilingual-mpnet-base-v2': หลายภาษา (768 มิติ)
            - 'all-MiniLM-L6-v2': เร็ว (384 มิติ)
        batch_size: จำนวนหนังที่ประมวลผลในครั้งเดียว
        use_gpu: ใช้ GPU ถ้ามี (เร็วกว่า)
    
    ส่งคืน:
        numpy array ของ embeddings
    """
    
    print("="*80)
    print("🎓 การสร้าง Embeddings ขั้นสูง - Neural Network & Deep Learning")
    print("="*80)
    print()
    
    # โหลดข้อมูลที่ประมวลผลแล้ว
    print("📥 กำลังโหลดข้อมูลหนังที่ประมวลผลแล้ว...")
    movies = pd.read_pickle('data/movies.pkl')
    print(f"✅ โหลดหนัง {len(movies):,} เรื่อง")
    
    # ตั้งค่า device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n🖥️  อุปกรณ์: {device.upper()}")
    
    # โหลด model
    print(f"\n🧠 กำลังโหลด Neural Network Model: {model_name}")
    print(f"   นี่คือ model แบบ Transformer ที่มีกลไก attention")
    model = SentenceTransformer(model_name, device=device)
    
    model_info = {
        'all-mpnet-base-v2': {
            'dims': 768,
            'params': '110M',
            'quality': '⭐⭐⭐⭐ ดีมาก (Old Standard)',
            'speed': '🐢 ปานกลาง'
        },
        'BAAI/bge-base-en-v1.5': {
            'dims': 768,
            'params': '110M',
            'quality': '⭐⭐⭐⭐⭐ ดีที่สุด (New SOTA)',
            'speed': '🚀 เร็ว'
        },
        'Alibaba-NLP/gte-large-en-v1.5': {
            'dims': 1024,
            'params': '434M',
            'quality': '⭐⭐⭐⭐⭐⭐ เทพเจ้า (Best Accuracy)',
            'speed': '🐢🐢 ช้ามาก'
        },
        'paraphrase-multilingual-mpnet-base-v2': {
            'dims': 768,
            'params': '278M',
            'quality': '⭐⭐⭐⭐ ดี + หลายภาษา',
            'speed': '🐢 ช้ากว่า'
        },
        'all-MiniLM-L6-v2': {
            'dims': 384,
            'params': '22M',
            'quality': '⭐⭐⭐ ดีพอใช้',
            'speed': '🚀🚀 เร็วมาก'
        }
    }
    
    if model_name in model_info:
        info = model_info[model_name]
        print(f"   📊 มิติ: {info['dims']}")
        print(f"   🔧 พารามิเตอร์: {info['params']}")
        print(f"   ✨ คุณภาพ: {info['quality']}")
        print(f"   ⚡ ความเร็ว: {info['speed']}")
    
    # สร้าง embeddings
    print(f"\n🎬 กำลังสร้าง embeddings สำหรับหนัง {len(movies):,} เรื่อง...")
    print(f"   ใช้คำอธิบายแบบครบถ้วนที่มี: ชื่อ, เนื้อเรื่อง, แนว, keywords, นักแสดง, ผู้กำกับ")
    print(f"   ขนาด batch: {batch_size}")
    print()
    
    # ประมวลผลแบบ batch พร้อม progress bar
    embeddings = []
    texts = movies['rich_description'].tolist()
    
    for i in tqdm(range(0, len(texts), batch_size), desc="กำลังประมวลผล batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization เพื่อความคล้ายคลึงที่ดีขึ้น
        )
        embeddings.append(batch_embeddings)
    
    # รวมทุก batches
    embeddings = np.vstack(embeddings)
    
    print(f"\n✅ สร้าง embeddings เสร็จสมบูรณ์!")
    print(f"   รูปแบบ: {embeddings.shape}")
    print(f"   ขนาด: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # บันทึก embeddings
    output_file = 'data/movie_embeddings.npy'
    np.save(output_file, embeddings)
    print(f"\n💾 บันทึก embeddings ที่: {output_file}")
    
    # บันทึกข้อมูล model
    model_info_file = 'data/model_info.txt'
    with open(model_info_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"มิติ: {embeddings.shape[1]}\n")
        f.write(f"หนัง: {len(movies):,}\n")
        f.write(f"อุปกรณ์: {device}\n")
    
    return embeddings

def compare_models():
    """
    สร้าง embeddings ด้วยทุก models เพื่อเปรียบเทียบ
    (ตัวเลือก - สำหรับการวิเคราะห์เชิงวิชาการ)
    """
    models = [
        'all-MiniLM-L6-v2',           # เร็วพื้นฐาน
        'all-mpnet-base-v2',          # คุณภาพดีที่สุด
    ]
    
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"กำลังทดสอบ model: {model_name}")
        print(f"{'='*80}\n")
        
        embeddings = generate_advanced_embeddings(
            model_name=model_name,
            batch_size=32
        )
        
        # บันทึกด้วยชื่อเฉพาะของ model
        output_file = f'data/embeddings_{model_name.replace("/", "_")}.npy'
        np.save(output_file, embeddings)
        print(f"บันทึกที่: {output_file}")

if __name__ == "__main__":
    # สร้างด้วย model ที่ดีที่สุด
    embeddings = generate_advanced_embeddings(
        model_name='BAAI/bge-base-en-v1.5',  # ⭐ New SOTA Model
        batch_size=32,
        use_gpu=False  # ตั้งเป็น True ถ้ามี CUDA GPU
    )
    
    print("\n" + "="*80)
    print("✨ พร้อมสำหรับระบบแนะนำขั้นสูง!")
    print("="*80)
    print("\n📝 ขั้นตอนถัดไป:")
    print("   1. รัน: python app.py")
    print("   2. ทดสอบ AI ที่ปรับปรุงด้วยคำค้นที่ซับซ้อน")
    print("   3. เปรียบเทียบผลลัพธ์กับข้อมูลหนัง 5,000 เรื่องเดิม")
