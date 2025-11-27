"""
ระบบประมวลผลข้อมูลหนังขั้นสูง
สำหรับวิชา Neural Network & Deep Learning

ฟีเจอร์:
- หนัง 45,000+ เรื่อง จาก Kaggle "The Movies Dataset"
- Metadata ครบถ้วน: keywords, นักแสดง, ทีมงาน, แนวหนัง
- การประมวลผลข้อความขั้นสูงสำหรับ embeddings ที่ดีขึ้น
- กรองคุณภาพเพื่อผลลัพธ์ที่ดีที่สุด
"""

import pandas as pd
import numpy as np
import json
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

def parse_json_column(x):
    """แปลงคอลัมน์ JSON อย่างปลอดภัย"""
    try:
        if pd.isna(x) or x == '':
            return []
        return literal_eval(x)
    except:
        return []

def extract_names(obj_list, key='name', max_items=5):
    """ดึงชื่อจาก list ของ dictionaries"""
    if not isinstance(obj_list, list):
        return []
    return [item[key] for item in obj_list[:max_items] if key in item]

def extract_director(crew_list):
    """ดึงชื่อผู้กำกับจากทีมงาน"""
    if not isinstance(crew_list, list):
        return ''
    directors = [person['name'] for person in crew_list if person.get('job') == 'Director']
    return directors[0] if directors else ''

def clean_text(text):
    """ทำความสะอาดและปรับข้อความให้เป็นมาตรฐาน"""
    if pd.isna(text):
        return ''
    return str(text).strip().lower()

def process_movies_dataset():
    """
    กระบวนการประมวลผลหลักสำหรับ The Movies Dataset
    
    ส่งคืน:
        DataFrame พร้อมข้อมูลหนังที่ประมวลผลแล้ว พร้อมสำหรับสร้าง embeddings
    """
    # --- โหลดและรวมข้อมูลทั้งสองแหล่ง ---
    import os
    import pickle
    
    all_movies = []
    
    # --- 1. โหลดข้อมูลใหม่จาก TMDB (2025) ---
    fresh_data_path = 'data/movies_fresh.pkl'
    if os.path.exists(fresh_data_path):
        print(f"🌟 กำลังโหลดข้อมูล TMDB ใหม่...")
        with open(fresh_data_path, 'rb') as f:
            fresh_data = pickle.load(f)
        
        tmdb_movies = pd.DataFrame(fresh_data)
        tmdb_movies['source'] = 'TMDB'  # Tag source
        print(f"✅ โหลด TMDB: {len(tmdb_movies):,} เรื่อง (ปี 2024-2025)")
        all_movies.append(tmdb_movies)
    else:
        print("⚠️ ไม่พบข้อมูล TMDB ใหม่")
    
    # --- 2. โหลดข้อมูลเก่าจาก Kaggle ---
    kaggle_path = 'data/movies_dataset/movies_metadata.csv'
    if os.path.exists(kaggle_path):
        print("🎬 กำลังโหลดข้อมูล Kaggle (16K+ เรื่อง)...")
        
        # โหลดข้อมูลหลัก
        kaggle_movies = pd.read_csv(kaggle_path, low_memory=False)
        
        # โหลดข้อมูลเพิ่มเติม
        keywords = pd.read_csv('data/movies_dataset/keywords.csv')
        credits = pd.read_csv('data/movies_dataset/credits.csv')
        
        print(f"📊 โหลด Kaggle: {len(kaggle_movies):,} เรื่อง")
        
        # --- การทำความสะอาดข้อมูล ---
        print("🧹 กำลังทำความสะอาด Kaggle...")
        
        # ลบ ID ที่ไม่ถูกต้อง
        kaggle_movies = kaggle_movies[kaggle_movies['id'].notna()]
        kaggle_movies['id'] = pd.to_numeric(kaggle_movies['id'], errors='coerce')
        kaggle_movies = kaggle_movies[kaggle_movies['id'].notna()]
        kaggle_movies['id'] = kaggle_movies['id'].astype(int)
        
        # แปลงคอลัมน์ JSON
        print("📦 กำลังแปลงคอลัมน์ JSON...")
        kaggle_movies['genres'] = kaggle_movies['genres'].apply(parse_json_column)
        kaggle_movies['production_companies'] = kaggle_movies['production_companies'].apply(parse_json_column)
        kaggle_movies['production_countries'] = kaggle_movies['production_countries'].apply(parse_json_column)
        kaggle_movies['spoken_languages'] = kaggle_movies['spoken_languages'].apply(parse_json_column)
        
        keywords['keywords'] = keywords['keywords'].apply(parse_json_column)
        credits['cast'] = credits['cast'].apply(parse_json_column)
        credits['crew'] = credits['crew'].apply(parse_json_column)
        
        # --- รวมชุดข้อมูล ---
        print("🔗 กำลังรวม metadata Kaggle...")
        kaggle_movies = kaggle_movies.merge(keywords, on='id', how='left')
        kaggle_movies = kaggle_movies.merge(credits, on='id', how='left')
        kaggle_movies['source'] = 'Kaggle'
        
        print(f"✅ ประมวลผล Kaggle เสร็จ: {len(kaggle_movies):,} เรื่อง")
        all_movies.append(kaggle_movies)
    else:
        print("⚠️ ไม่พบข้อมูล Kaggle")
    
    # --- 3. รวมและลบซ้ำ ---
    if len(all_movies) == 0:
        raise FileNotFoundError("ไม่พบข้อมูลหนังเลย! ต้องมีอย่างน้อย 1 แหล่งข้อมูล")
    
    print(f"\n🔄 กำลังรวมข้อมูลจาก {len(all_movies)} แหล่ง...")
    movies = pd.concat(all_movies, ignore_index=True)
    print(f"📊 รวมทั้งหมด: {len(movies):,} เรื่อง (ก่อนลบซ้ำ)")
    
    # ลบซ้ำโดยใช้ title (case-insensitive) - เก็บ TMDB ถ้ามี
    movies['title_lower'] = movies['title'].str.lower().str.strip()
    movies = movies.sort_values('source', ascending=False)  # TMDB > Kaggle (T > K alphabetically)
    movies = movies.drop_duplicates(subset='title_lower', keep='first')
    movies = movies.drop('title_lower', axis=1)
    
    print(f"✅ หลังลบซ้ำ: {len(movies):,} เรื่อง")
    
    # --- ดึงฟีเจอร์ ---
    print("🎯 กำลังดึงฟีเจอร์...")
    
    # ดึงชื่อจากโครงสร้างแบบซ้อน
    movies['genre_names'] = movies['genres'].apply(lambda x: extract_names(x, 'name', 10))
    movies['keyword_names'] = movies['keywords'].apply(lambda x: extract_names(x, 'name', 15))
    movies['cast_names'] = movies['cast'].apply(lambda x: extract_names(x, 'name', 10))
    movies['director'] = movies['crew'].apply(extract_director)
    movies['production_company_names'] = movies['production_companies'].apply(lambda x: extract_names(x, 'name', 3))
    
    # --- การกรองคุณภาพ ---
    print("✨ กำลังกรองคุณภาพ...")
    
    # กรอง: ต้องมีชื่อและเรื่องย่อ
    movies = movies[movies['title'].notna() & movies['overview'].notna()]
    
    # กรอง: หนังภาษาอังกฤษเท่านั้น (ปิดไว้เพื่อรักษา TMDB + Kaggle ทั้งหมด)
    # if 'original_language' in movies.columns:
    #     movies = movies[movies['original_language'] == 'en']
    # else:
    #     print("⚠️ ไม่พบคอลัมน์ original_language - ข้ามการกรองภาษา")
    
    # กรอง: ต้องมีวันที่ออกฉายที่ถูกต้อง
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies = movies[movies['release_date'].notna()]
    movies['release_year'] = movies['release_date'].dt.year
    
    # กรอง: จำนวนโหวตขั้นต่ำ (ตัวบ่งชี้คุณภาพ)
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce')
    movies = movies[movies['vote_count'] >= 10]  # อย่างน้อย 10 โหวต
    
    # กรอง: คะแนนที่ถูกต้อง
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce')
    movies = movies[movies['vote_average'] > 0]
    
    # กรอง: ต้องมีอย่างน้อย 1 แนว
    movies = movies[movies['genre_names'].apply(len) > 0]
    
    # --- สร้างคำอธิบายแบบครบถ้วนสำหรับ AI ---
    print("🤖 กำลังสร้างคำอธิบายที่ปรับให้เหมาะกับ AI...")
    
    def create_rich_description(row):
        """
        สร้างคำอธิบายข้อความแบบครบถ้วนสำหรับ neural network embedding
        
        รวมฟีเจอร์หลายอย่างเพื่อให้ AI มี context สูงสุด:
        - ชื่อและเรื่องย่อ (เนื้อหาหลัก)
        - แนวหนัง (การจัดหมวดหมู่)
        - Keywords (ธีมและหัวข้อ)
        - นักแสดงและผู้กำกับ (ตัวบ่งชี้สไตล์)
        - บริษัทผลิต (สัญญาณคุณภาพ)
        """
        parts = []
        
        # ชื่อ (สำคัญ! - ใส่ 2 รอบเพื่อเน้น)
        parts.append(f"Title: {row['title']}")
        parts.append(f"{row['title']}")
        
        # เรื่องย่อ/พล็อต
        if pd.notna(row['overview']) and str(row['overview']).strip():
            parts.append(f"Plot: {row['overview']}")
        
        # แนวหนัง
        if row['genre_names']:
            parts.append(f"Genres: {', '.join(row['genre_names'])}")
        
        # Keywords (สำคัญมากสำหรับการค้นหาเชิงความหมาย!)
        if row['keyword_names']:
            parts.append(f"Keywords: {', '.join(row['keyword_names'])}")
            # ใส่ keywords สำคัญซ้ำเพื่อเน้น
            parts.append(f"{', '.join(row['keyword_names'][:5])}")
        
        # ผู้กำกับ
        if row['director']:
            parts.append(f"Director: {row['director']}")
        
        # นักแสดงหลัก
        if row['cast_names']:
            parts.append(f"Cast: {', '.join(row['cast_names'][:5])}")
        
        # บริษัทผลิต
        if row['production_company_names']:
            parts.append(f"Studio: {', '.join(row['production_company_names'])}")
            
        # ข้อมูลเพิ่มเติมเพื่อช่วยในการแยกแยะ
        parts.append(f"Year: {row['release_year']:.0f}")
        parts.append(f"Rating: {row['vote_average']:.1f}")
        
        return ' | '.join(parts)
    
    movies['rich_description'] = movies.apply(create_rich_description, axis=1)
    
    # --- เตรียมชุดข้อมูลสุดท้าย ---
    print("📋 กำลังเตรียมชุดข้อมูลสุดท้าย...")
    
    # เลือกและเปลี่ยนชื่อคอลัมน์
    final_columns = {
        'id': 'movie_id',
        'title': 'title',
        'overview': 'overview',
        'rich_description': 'rich_description',
        'genre_names': 'genres',
        'keyword_names': 'keywords',
        'cast_names': 'cast',
        'director': 'director',
        'release_year': 'release_year',
        'vote_average': 'vote_average',
        'vote_count': 'vote_count',
        'popularity': 'popularity',
        'runtime': 'runtime',
        'budget': 'budget',
        'revenue': 'revenue'
    }
    
    # Ensure all columns exist
    for col in final_columns.keys():
        if col not in movies.columns:
            movies[col] = 0
            
    movies_processed = movies[list(final_columns.keys())].copy()
    movies_processed.rename(columns=final_columns, inplace=True)
    
    # แปลงคอลัมน์ตัวเลข
    numeric_cols = ['popularity', 'runtime', 'budget', 'revenue']
    for col in numeric_cols:
        movies_processed[col] = pd.to_numeric(movies_processed[col], errors='coerce').fillna(0)
    
    # เรียงตามความนิยม (หนังดีที่สุดก่อน)
    movies_processed = movies_processed.sort_values('popularity', ascending=False)
    
    # รีเซ็ต index
    movies_processed.reset_index(drop=True, inplace=True)
    
    print(f"\n✅ การประมวลผลเสร็จสมบูรณ์!")
    print(f"📊 ชุดข้อมูลสุดท้าย: {len(movies_processed):,} หนังคุณภาพสูง")
    print(f"📅 ช่วงปี: {movies_processed['release_year'].min():.0f} - {movies_processed['release_year'].max():.0f}")
    print(f"⭐ คะแนนเฉลี่ย: {movies_processed['vote_average'].mean():.2f}")
    
    # บันทึกข้อมูลที่ประมวลผลแล้ว
    output_file = 'data/movies.pkl'
    movies_processed.to_pickle(output_file)
    print(f"\n💾 บันทึกที่: {output_file}")
    
    # แสดงตัวอย่าง
    print("\n🎬 หนังตัวอย่าง:")
    sample = movies_processed[['title', 'release_year', 'vote_average', 'genres']].head(10)
    for idx, row in sample.iterrows():
        print(f"  {idx+1}. {row['title']} ({row['release_year']:.0f}) - ⭐{row['vote_average']:.1f} - {', '.join(row['genres'][:3])}")
    
    return movies_processed

if __name__ == "__main__":
    print("="*80)
    print("🎓 ระบบแนะนำหนังขั้นสูง - Neural Network & Deep Learning")
    print("="*80)
    print()
    
    movies_df = process_movies_dataset()
    
    print("\n" + "="*80)
    print("✨ พร้อมสำหรับการสร้าง Embeddings ขั้นสูง!")
    print("="*80)
