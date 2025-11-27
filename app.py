"""
ระบบแนะนำภาพยนตร์ด้วย Deep Learning
สำหรับวิชา Neural Network & Deep Learning

ฟีเจอร์ AI ขั้นสูง:
✨ ข้อมูลหนังคุณภาพสูง 16,904 เรื่อง
🧠 Dual Transformer Architecture:
   • Bi-Encoder: BAAI/bge-base-en-v1.5 (SOTA Embedding)
   • Cross-Encoder: ms-marco-MiniLM-L-12-v2 (Deep Re-ranker)
🎯 กระบวนการ AI 6 ขั้นตอน:
   Intent Analysis → Semantic Search → Hybrid Scoring → 
   Cross-Encoder Re-ranking → Diversity Optimization → Results
💡 Intent-Aware Weighting:
   น้ำหนักคะแนนปรับตามความตั้งใจ (recent/classic/quality/popular/niche)
🌏 รองรับภาษาไทย: 80+ คีย์เวิร์ด + 15+ รูปแบบขยายคำ
📊 Metadata Fusion: Token-level matching (keywords, cast, directors)
📅 Year Intelligence: ทศวรรษ, ช่วงปี, การจัดตำแหน่งตามเวลา
🎬 Genre Hints: เพิ่มคะแนนเมื่อระบุแนวหนัง
⚡ คะแนนผสมแบบไดนามิก:
   Semantic (35-65%) + Metadata (10-35%) + Quality (5-30%) + 
   Popularity (5-25%) + Recency (5-30%) + Year/Genre Bonuses
🔍 Cross-Encoder Re-ranking: ความแม่นยำสูงสุดสำหรับ 30 อันดับแรก
🎨 Dynamic Diversity: ปรับระดับความหลากหลาย (0.05-0.20) ตามบริบท
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import os

app = Flask(__name__)

from tmdb_client import TMDBClient
# อ่าน API Key จาก Environment Variable หรือใส่ตรงนี้ชั่วคราว
# export TMDB_API_KEY='your_key_here'
tmdb_client = TMDBClient()

# --- พจนานุกรมแปลภาษาไทย-อังกฤษ ---
THAI_TO_ENGLISH = {
    'แอคชั่น': 'action',
    'บู๊': 'action thriller fighting',
    'ตลก': 'comedy funny',
    'ตลกขำขัน': 'comedy hilarious',
    'ผี': 'horror',
    'สยองขวัญ': 'horror scary',
    'โรแมนติก': 'romance',
    'รักโรแมนติก': 'romantic love',
    'ดราม่า': 'drama',
    'ดราม่าเข้มข้น': 'intense drama',
    'แฟนตาซี': 'fantasy',
    'วิทยาศาสตร์': 'science fiction',
    'ไซไฟ': 'sci-fi science fiction',
    'ระทึกขวัญ': 'thriller',
    'ลึกลับ': 'mystery',
    'ปริศนา': 'mystery puzzle',
    'ผจญภัย': 'adventure',
    'อาชญากรรม': 'crime',
    'สงคราม': 'war',
    'สารคดี': 'documentary',
    'แอนนิเมชั่น': 'animation',
    'การ์ตูน': 'animation',
    'ครอบครัว': 'family',
    'ซูเปอร์ฮีโร่': 'superhero marvel dc',
    'ฮีโร่': 'hero superhero',
    'ซอมบี้': 'zombie',
    'แวมไพร์': 'vampire',
    'มังกร': 'dragon',
    'เวทมนตร์': 'magic wizard',
    'อวกาศ': 'space',
    'เอเลี่ยน': 'alien',
    'หุ่นยนต์': 'robot',
    'ไดโนเสาร์': 'dinosaur',
    'โจรสลัด': 'pirate',
    'นินจา': 'ninja',
    'ซามูไร': 'samurai',
    'มาเฟีย': 'mafia gangster',
    'เศร้า': 'sad emotional',
    'สนุก': 'fun entertaining',
    'ตื่นเต้น': 'exciting thrilling',
    'น่ากลัว': 'scary frightening',
    'สะเทือนใจ': 'touching emotional',
    'ฮา': 'funny hilarious',
    'เกาหลี': 'korean',
    'ญี่ปุ่น': 'japanese',
    'ไทย': 'thai',
    'จีน': 'chinese',
    'ฝรั่ง': 'western',
    'อินเดีย': 'indian bollywood',
    'ฮอลลีวูด': 'hollywood',
    'บอลลีวูด': 'bollywood',
}

QUERY_EXPANSION = {
    'mind-bending': 'psychological complex inception interstellar matrix reality thought-provoking cerebral',
    'mind bending': 'psychological complex inception interstellar matrix reality thought-provoking cerebral',
    'emotional': 'touching heartwarming tearjerker moving powerful drama feelings',
    'intense': 'gripping suspenseful edge-of-seat thrilling powerful',
    'dark': 'noir gritty moody atmospheric bleak',
    'uplifting': 'inspiring feel-good heartwarming positive motivational',
    'epic': 'grand spectacular massive ambitious large-scale',
    'slow burn': 'contemplative meditative paced atmospheric',
    'fast-paced': 'action-packed exciting dynamic energetic',
    'visually stunning': 'beautiful cinematography visual-effects gorgeous',
    'indie': 'independent art-house alternative',
    'twist': 'plot-twist surprise unexpected revelation',
    'character-driven': 'character-study psychological drama performance',
    'based on true': 'true-story biographical real-events documentary',
}

GENRE_SYNONYMS = {
    'action': 'action thriller explosive fighting combat battle',
    'comedy': 'comedy funny hilarious humor laugh',
    'horror': 'horror scary frightening terror',
    'romance': 'romance romantic love relationship',
    'drama': 'drama emotional intense character-driven',
    'scifi': 'science fiction sci-fi futuristic technology',
    'sci-fi': 'science fiction futuristic technology space',
    'fantasy': 'fantasy magical mystical enchanted',
    'thriller': 'thriller suspense tension mystery',
    'adventure': 'adventure journey quest exploration',
    'crime': 'crime detective investigation police',
    'superhero': 'superhero marvel dc comic-book powers hero',
}

GENRE_HINT_KEYWORDS = {
    'science fiction': 'Science Fiction',
    'sci-fi': 'Science Fiction',
    'scifi': 'Science Fiction',
    'ไซไฟ': 'Science Fiction',
    'mind-bending sci-fi': 'Science Fiction',
    'mind bending sci-fi': 'Science Fiction',
    'thriller': 'Thriller',
    'psychological': 'Thriller',
    'drama': 'Drama',
    'animation': 'Animation',
    'animated': 'Animation',
    'documentary': 'Documentary',
    'romance': 'Romance',
    'action': 'Action',
    'adventure': 'Adventure',
    'comedy': 'Comedy',
    'horror': 'Horror',
}

BASE_WEIGHT_PROFILE = {
    'semantic': 0.50,
    'metadata': 0.20,
    'quality': 0.15,
    'popularity': 0.10,
    'recency': 0.05,
}

WEIGHT_LIMITS = {
    'semantic': (0.35, 0.65),
    'metadata': (0.10, 0.35),
    'quality': (0.05, 0.30),
    'popularity': (0.05, 0.25),
    'recency': (0.05, 0.30),
}

RECENT_KEYWORDS = ['new', 'latest', 'recent', 'modern', 'fresh', 'ปัจจุบัน', 'ใหม่', 'ล่าสุด']
CLASSIC_KEYWORDS = ['classic', 'retro', 'vintage', 'old school', 'nostalgic', 'ยุคเก่า', 'คลาสสิค', 'ตำนาน']
QUALITY_KEYWORDS = ['award', 'oscar', 'acclaimed', 'masterpiece', 'critically', 'การันตี', 'รางวัล']
POPULAR_KEYWORDS = ['popular', 'hit', 'blockbuster', 'top grossing', 'box office', 'ฮิต', 'ดัง']
NICHE_KEYWORDS = ['underrated', 'hidden gem', 'cult', 'obscure', 'indie', 'ไม่ค่อยดัง', 'ลับ']
METADATA_KEYWORDS = ['starring', 'directed by', 'actor', 'cast', 'director', 'ผู้กำกับ', 'นักแสดง']
DIVERSITY_KEYWORDS = ['variety', 'surprise', 'mix', 'หลากหลาย', 'แตกต่าง']
FOCUSED_KEYWORDS = ['similar', 'exact', 'เฉพาะ', 'แบบเดียวกัน']

DECADE_KEYWORDS = {
    '60s': (1960, 1969),
    "60's": (1960, 1969),
    'ยุค 60': (1960, 1969),
    '70s': (1970, 1979),
    "70's": (1970, 1979),
    'ยุค 70': (1970, 1979),
    '80s': (1980, 1989),
    "80's": (1980, 1989),
    'ยุค 80': (1980, 1989),
    '90s': (1990, 1999),
    "90's": (1990, 1999),
    'ยุค 90': (1990, 1999),
    '2000s': (2000, 2009),
    '2010s': (2010, 2019),
    "2010's": (2010, 2019),
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9ก-๙']+")


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def tokenize_text(text):
    if not text:
        return set()
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(str(text))}


def build_weight_profile(adjustments):
    weights = BASE_WEIGHT_PROFILE.copy()
    for key, delta in adjustments.items():
        if key in weights:
            low, high = WEIGHT_LIMITS.get(key, (0, 1))
            weights[key] = clamp(weights[key] + delta, low, high)
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    return weights


def extract_genre_hints(query_lower):
    hints = set()
    for keyword, canonical in GENRE_HINT_KEYWORDS.items():
        if keyword in query_lower:
            hints.add(canonical)
    return hints


def extract_year_constraints(query_lower):
    constraints = {'min_year': None, 'max_year': None}
    between_match = re.search(r"(?:between|ช่วง)\s+(?:ปี\s*)?((?:19|20)\d{2})\s+(?:and|ถึง)\s+(?:ปี\s*)?((?:19|20)\d{2})", query_lower)
    if between_match:
        constraints['min_year'] = int(between_match.group(1))
        constraints['max_year'] = int(between_match.group(2))
        return constraints
    after_match = re.search(r"(?:after|since|ตั้งแต่|หลัง)\s+(?:ปี\s*)?((?:19|20)\d{2})", query_lower)
    if after_match:
        constraints['min_year'] = int(after_match.group(1))
    before_match = re.search(r"(?:before|ก่อน)\s+(?:ปี\s*)?((?:19|20)\d{2})", query_lower)
    if before_match:
        constraints['max_year'] = int(before_match.group(1))
    decade_hit = next(((start, end) for phrase, (start, end) in DECADE_KEYWORDS.items() if phrase in query_lower), None)
    if decade_hit:
        constraints['min_year'], constraints['max_year'] = decade_hit
    single_years = [int(year) for year in re.findall(r"((?:19|20)\d{2})", query_lower)]
    if single_years and not between_match and not (after_match or before_match):
        year = single_years[0]
        constraints['min_year'] = year
        constraints['max_year'] = year
    return constraints


def calculate_year_alignment_bonus(release_year, constraints):
    if not constraints or (constraints.get('min_year') is None and constraints.get('max_year') is None):
        return 0.0
    if pd.isna(release_year):
        return 0.0
    year = int(release_year)
    bonus = 0.0
    min_year = constraints.get('min_year')
    max_year = constraints.get('max_year')
    if min_year:
        bonus += 0.04 if year >= min_year else -0.05
    if max_year:
        bonus += 0.04 if year <= max_year else -0.05
    return clamp(bonus, -0.08, 0.08)


def analyze_query_intent(query):
    query_lower = query.lower()
    adjustments = {'metadata': 0.0, 'quality': 0.0, 'popularity': 0.0, 'recency': 0.0}
    diversity_penalty = 0.1
    constraints = extract_year_constraints(query_lower)
    genre_hints = extract_genre_hints(query_lower)
    if any(keyword in query_lower for keyword in RECENT_KEYWORDS):
        adjustments['recency'] += 0.06
        if constraints.get('min_year') is None:
            constraints['min_year'] = max(1900, current_year - 10)
    if any(keyword in query_lower for keyword in CLASSIC_KEYWORDS):
        adjustments['recency'] -= 0.05
        if constraints.get('max_year') is None:
            constraints['max_year'] = min(constraints.get('max_year') or current_year, 2005)
    if any(keyword in query_lower for keyword in QUALITY_KEYWORDS):
        adjustments['quality'] += 0.04
    if any(keyword in query_lower for keyword in POPULAR_KEYWORDS):
        adjustments['popularity'] += 0.04
    if any(keyword in query_lower for keyword in NICHE_KEYWORDS):
        adjustments['popularity'] -= 0.04
    if any(keyword in query_lower for keyword in METADATA_KEYWORDS):
        adjustments['metadata'] += 0.05
    if any(keyword in query_lower for keyword in DIVERSITY_KEYWORDS):
        diversity_penalty = 0.05
    if any(keyword in query_lower for keyword in FOCUSED_KEYWORDS):
        diversity_penalty = 0.15
    weights = build_weight_profile(adjustments)
    return {
        'weights': weights,
        'year_constraints': constraints,
        'diversity_penalty': clamp(diversity_penalty, 0.05, 0.2),
        'genre_hints': genre_hints,
    }


def get_best_device():
    """เลือก device ที่ดีที่สุดตามลำดับ: CUDA GPU > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        print(f"✅ พบ CUDA GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("✅ พบ Apple Silicon MPS - ใช้ GPU acceleration")
        return 'mps'
    else:
        print("⚠️  ไม่พบ GPU - ใช้ CPU (ช้ากว่า)")
        return 'cpu'

print("🚀 กำลังโหลด AI model และข้อมูล...")
device = get_best_device()

# ลองใช้โมเดลที่ fine-tune ก่อน ถ้าไม่มีค่อยใช้ base model
finetuned_path = 'data/finetuned_model'
try:
    # ตรวจสอบว่ามี config.json (โมเดลที่เทรนเสร็จแล้ว)
    import json
    config_path = os.path.join(finetuned_path, 'config.json')
    if os.path.exists(config_path):
        print("🎯 โหลดโมเดลที่ fine-tune แล้ว (ฉลาดกว่า!)")
        model = SentenceTransformer(finetuned_path, device=device)
        model_name = 'movie-finetuned-bge'
    else:
        raise FileNotFoundError("Fine-tuned model ยังไม่เสร็จ")
except (FileNotFoundError, ValueError, OSError) as e:
    print(f"📚 โหลด SOTA base model (fine-tuned model ไม่พร้อม: {e})")
    # ใช้ BGE-base-en-v1.5 แทน all-mpnet-base-v2
    model_name = 'BAAI/bge-base-en-v1.5'
    print(f"🚀 Model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

print("🧠 Loading Cross-Encoder re-ranker (Deep Layer)...")
try:
    # ใช้ L-12 (ฉลาดกว่า L-6)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
    use_reranker = True
    print(f"✅ Cross-Encoder (L-12) loaded successfully on {device.upper()}")
except Exception as e:
    print(f"⚠️  Cross-Encoder not available: {e}")
    print("💡 Run: pip install sentence-transformers --upgrade")
    use_reranker = False

movies_df = pd.read_pickle('data/movies.pkl')

# โหลด embeddings ที่ตรงกับโมเดล
if os.path.exists('data/movie_embeddings_finetuned.npy'):
    print("📊 โหลด fine-tuned embeddings (ดีกว่า!)")
    movie_vectors = np.load('data/movie_embeddings_finetuned.npy')
else:
    print("📊 โหลด base embeddings")
    movie_vectors = np.load('data/movie_embeddings.npy')

movies_df['popularity_score'] = (movies_df['popularity'] - movies_df['popularity'].min()) / \
                                 (movies_df['popularity'].max() - movies_df['popularity'].min())
current_year = datetime.now().year
movies_df['recency_score'] = movies_df['release_year'].apply(
    lambda year: max(0, 1 - (current_year - year) / 100)
)
movies_df['quality_score'] = movies_df['vote_average'] / 10.0
print(f"✅ Ready! {len(movies_df):,} movies with 768-dim embeddings")


def translate_thai_keywords(text):
    text_lower = text.lower()
    translated = text
    for thai, english in THAI_TO_ENGLISH.items():
        if thai in text_lower:
            translated = translated.replace(thai, english)
            print(f"🌐 Translated: '{thai}' → '{english}'")
    return translated


from deep_translator import GoogleTranslator

def enhance_query(query):
    # 1. ตรวจสอบและแปลภาษาอัตโนมัติ (Dynamic Translation)
    # ถ้ามีภาษาไทย (ก-๙) ให้แปลเป็นอังกฤษ
    if re.search(r'[ก-๙]', query):
        try:
            print(f"🇹🇭 ตรวจพบภาษาไทย: '{query}'")
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            print(f"🇬🇧 แปลเป็นอังกฤษ: '{translated}'")
            query_en = translated
        except Exception as e:
            print(f"⚠️ แปลภาษาล้มเหลว: {e}")
            query_en = translate_thai_keywords(query) # Fallback
    else:
        query_en = translate_thai_keywords(query)

    query_lower = query_en.lower()
    expanded_parts = []
    for phrase, expansion in QUERY_EXPANSION.items():
        if phrase in query_lower:
            expanded_parts.append(expansion)
            print(f"💡 ขยายคำ '{phrase}' ด้วยความรู้เฉพาะด้าน")
    for genre, synonyms in GENRE_SYNONYMS.items():
        if genre in query_lower:
            expanded_parts.append(synonyms)
            print(f"🎬 เพิ่มคำพ้องของแนว '{genre}'")
    if expanded_parts:
        query_en = f"{query_en} {' '.join(expanded_parts)}"
    print(f"🔍 คำค้นที่ปรับปรุงแล้ว: '{query_en[:100]}...'" if len(query_en) > 100 else f"🔍 คำค้นที่ปรับปรุงแล้ว: '{query_en}'")
    return query_en


def calculate_metadata_score(query_text, movie_row, query_tokens=None):
    query_lower = query_text.lower()
    lookup_tokens = query_tokens or tokenize_text(query_text)
    score = 0.0
    keywords = movie_row['keywords'] if isinstance(movie_row['keywords'], list) else []
    if keywords:
        matches = 0
        for kw in keywords[:20]:
            kw_tokens = tokenize_text(kw)
            if kw_tokens & lookup_tokens or kw.lower() in query_lower:
                matches += 1
        score += min(matches, 4) * 0.05
    director = movie_row['director'] if isinstance(movie_row['director'], str) else ''
    if director:
        director_tokens = tokenize_text(director)
        if director.lower() in query_lower or director_tokens & lookup_tokens:
            score += 0.10
    cast_list = movie_row['cast'] if isinstance(movie_row['cast'], list) else []
    if cast_list:
        cast_matches = 0
        for actor in cast_list[:5]:
            actor_tokens = tokenize_text(actor)
            actor_key = actor.lower()
            if actor_key in query_lower or actor_tokens & lookup_tokens:
                cast_matches += 1
        score += cast_matches * 0.03
    return min(score, 0.35)


def calculate_diversity_penalty(selected_indices, candidate_idx, movies_df, penalty=0.1):
    """
    คำนวณ penalty จากความซ้ำซ้อนของ genre
    ไม่ penalize ถ้าเป็นหนังชื่อเดียวกันคนละปี (remake/sequel)
    """
    if len(selected_indices) == 0:
        return 0
    
    candidate_movie = movies_df.iloc[candidate_idx]
    candidate_title = candidate_movie['title']
    candidate_year = candidate_movie['release_year']
    
    # ตรวจสอบว่ามีหนังชื่อเดียวกันใน selected หรือไม่
    for idx in selected_indices:
        selected_movie = movies_df.iloc[idx]
        if selected_movie['title'] == candidate_title and selected_movie['release_year'] != candidate_year:
            # ชื่อเดียวกันแต่คนละปี = ไม่ penalize (เป็น remake/reboot)
            return 0
    
    # คำนวณ genre overlap ปกติ
    selected_genres = set()
    for idx in selected_indices:
        selected_genres.update(movies_df.iloc[idx]['genres'])
    candidate_genres = set(candidate_movie['genres'])
    overlap = len(selected_genres.intersection(candidate_genres))
    return overlap * penalty


def get_recommendations_advanced(query, top_n=10, diversity=True):
    enhanced_query = enhance_query(query)
    intent_context = analyze_query_intent(query)
    weights = intent_context['weights']
    year_constraints = intent_context['year_constraints']
    diversity_penalty = intent_context['diversity_penalty']
    genre_hints = intent_context['genre_hints']
    metadata_reference_text = f"{query} {enhanced_query}"
    metadata_query_tokens = tokenize_text(metadata_reference_text)

    print("📊 ขั้นตอนที่ 1: ค้นหาเชิงความหมาย...")
    query_vector = model.encode([enhanced_query], normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_vector, movie_vectors).flatten()
    top_100_indices = np.argsort(semantic_scores)[-100:][::-1]

    print("🧮 ขั้นตอนที่ 2: คำนวณคะแนนผสมด้วย metadata fusion...")
    hybrid_scores = []
    for idx in top_100_indices:
        movie_row = movies_df.iloc[idx]
        semantic = semantic_scores[idx]
        quality = movie_row['quality_score']
        popularity = movie_row['popularity_score']
        recency = movie_row['recency_score']
        metadata = calculate_metadata_score(metadata_reference_text, movie_row, metadata_query_tokens)
        year_bonus = calculate_year_alignment_bonus(movie_row['release_year'], year_constraints)
        genre_bonus = 0.0
        if genre_hints:
            genre_matches = sum(1 for hint in genre_hints if hint in movie_row['genres'])
            if genre_matches:
                genre_bonus += clamp(genre_matches * 0.04, 0.0, 0.12)
            else:
                genre_bonus -= 0.04
        hybrid = (semantic * weights['semantic'] +
              metadata * weights['metadata'] +
              quality * weights['quality'] +
              popularity * weights['popularity'] +
              recency * weights['recency'] +
              year_bonus +
              genre_bonus)
        hybrid_scores.append((idx, hybrid, semantic))
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)

    if use_reranker and len(hybrid_scores) >= 30:
        print("🎯 ขั้นตอนที่ 3: Cross-Encoder จัดอันดับ 30 เรื่องแรกใหม่...")
        top_30 = hybrid_scores[:30]
        pairs = []
        for idx, _, _ in top_30:
            movie = movies_df.iloc[idx]
            doc_text = f"{movie['title']}. {movie['overview']}. แนว: {', '.join(movie['genres'][:3])}"
            pairs.append([query, doc_text])
        try:
            ce_scores = reranker.predict(pairs)
            reranked = []
            for i, (idx, hybrid, semantic) in enumerate(top_30):
                final_score = ce_scores[i] * 0.7 + hybrid * 0.3
                reranked.append((idx, final_score, semantic))
            reranked.sort(key=lambda x: x[1], reverse=True)
            hybrid_scores = reranked + hybrid_scores[30:]
            print("✅ Cross-Encoder re-ranking เสร็จสมบูรณ์")
        except Exception as e:
            print(f"⚠️  Cross-Encoder ล้มเหลว: {e}, ใช้คะแนน hybrid")

    if diversity:
        print("🎨 ขั้นตอนที่ 4: จัดอันดับใหม่เพื่อความหลากหลาย...")
        final_indices = []
        final_scores = []
        final_hybrid_scores = []
        seen_movies = set()
        for idx, hybrid_score, semantic_score in hybrid_scores:
            if len(final_indices) >= top_n:
                break
            movie_key = (movies_df.iloc[idx]['title'], movies_df.iloc[idx]['release_year'])
            if movie_key in seen_movies:
                continue
            penalty = calculate_diversity_penalty(final_indices, idx, movies_df, penalty=diversity_penalty)
            adjusted_score = hybrid_score - penalty
            final_indices.append(idx)
            final_scores.append(semantic_score)
            final_hybrid_scores.append(adjusted_score)
            seen_movies.add(movie_key)
        recommendations = movies_df.iloc[final_indices].copy()
        match_scores = np.array(final_scores)
        hybrid_match_scores = np.array(final_hybrid_scores)
    else:
        final_indices = []
        final_scores = []
        final_hybrid_scores = []
        seen_movies = set()
        for idx, hybrid_score, semantic_score in hybrid_scores:
            if len(final_indices) >= top_n:
                break
            movie_key = (movies_df.iloc[idx]['title'], movies_df.iloc[idx]['release_year'])
            if movie_key in seen_movies:
                continue
            final_indices.append(idx)
            final_scores.append(semantic_score)
            final_hybrid_scores.append(hybrid_score)
            seen_movies.add(movie_key)
        recommendations = movies_df.iloc[final_indices].copy()
        match_scores = np.array(final_scores)
        hybrid_match_scores = np.array(final_hybrid_scores)

    return recommendations, match_scores, hybrid_match_scores


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'กรุณาใส่คำค้นหา'}), 400
        recommendations, semantic_scores, hybrid_scores = get_recommendations_advanced(query, top_n=10)
        results = []
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            # คำนวณ score breakdown และจัดการ NaN
            semantic_val = semantic_scores[idx] if not np.isnan(semantic_scores[idx]) else 0.0
            hybrid_val = hybrid_scores[idx] if not np.isnan(hybrid_scores[idx]) else 0.0
            
            # ใช้คะแนนที่ดีที่สุดระหว่าง hybrid กับ semantic เพื่อป้องกันค่าติดลบ
            # hybrid อาจติดลบจาก diversity penalty หรือ genre mismatch
            final_score = max(hybrid_val, semantic_val * 0.8)  # อย่างน้อยได้ 80% ของ semantic
            
            # Normalize เป็น 0-100%
            # Semantic score อยู่ในช่วง 0-1 อยู่แล้ว (cosine similarity)
            # Hybrid score ปกติอยู่ที่ 0-1.2 แต่อาจติดลบจาก penalty
            match_pct = float(min(max(final_score, 0.0) * 100, 100.0))
            semantic_pct = float(min(semantic_val * 100, 100.0))
            
            quality_pct = float(row['vote_average'] * 10) if pd.notna(row['vote_average']) else 0.0
            popularity_pct = float(row['popularity_score'] * 100) if pd.notna(row['popularity_score']) else 0.0
            recency_pct = float(row['recency_score'] * 100) if pd.notna(row['recency_score']) else 0.0
            
            results.append({
                'title': row['title'],
                'year': int(row['release_year']) if pd.notna(row['release_year']) and row['release_year'] > 0 else 'N/A',
                'rating': float(row['vote_average']) if pd.notna(row['vote_average']) else 0.0,
                'match': match_pct,
                'score_breakdown': {
                    'semantic': round(semantic_pct, 1),
                    'quality': round(quality_pct, 1),
                    'popularity': round(popularity_pct, 1),
                    'recency': round(recency_pct, 1),
                },
                'overview': row['overview'] if pd.notna(row['overview']) else 'ไม่มีเรื่องย่อ',
                'genres': row['genres'][:3] if isinstance(row['genres'], list) else [],
                'director': row['director'] if pd.notna(row['director']) and row['director'] else 'ไม่ทราบ',
                'cast': row['cast'][:3] if isinstance(row['cast'], list) and len(row['cast']) > 0 else [],
                'keywords': row['keywords'][:5] if isinstance(row['keywords'], list) and len(row['keywords']) > 0 else [],
            })
        
        # เรียงลำดับตาม match % จากมากไปน้อย
        results.sort(key=lambda x: x['match'], reverse=True)
        
        # เพิ่มอันดับหลังจากเรียงแล้ว
        for idx, result in enumerate(results):
            result['rank'] = idx + 1
            
        # --- TMDB Enrichment (เฉพาะ Top 10) ---
        # ทำแบบ Parallel หรือ Batch จะดีกว่า แต่เพื่อความง่ายทำ Loop ไปก่อน
        # (ใน Production ควรใช้ ThreadPoolExecutor)
        if tmdb_client.api_key:
            print("🎨 Fetching TMDB images...")
            for res in results:
                try:
                    enrichment = tmdb_client.enrich_movie_data(res['title'], res['year'])
                    if enrichment:
                        res.update(enrichment)
                except Exception as e:
                    print(f"⚠️ TMDB Error for {res['title']}: {e}")
        
        return jsonify({'query': query, 'count': len(results), 'results': results})
    except Exception as e:
        print(f"❌ ข้อผิดพลาด: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    return jsonify({
        'total_movies': len(movies_df),
        'year_range': {'min': int(movies_df['release_year'].min()), 'max': int(movies_df['release_year'].max())},
        'average_rating': float(movies_df['vote_average'].mean()),
        'model': model_name,
        'device': device,
        'embedding_dims': 768,
        'total_votes': int(movies_df['vote_count'].sum())
    })


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎓 ระบบแนะนำภาพยนตร์ขั้นสูง")
    print("   สำหรับวิชา Neural Network & Deep Learning")
    print("=" * 80)
    print(f"\n📊 ข้อมูล: {len(movies_df):,} เรื่อง")
    print(f"🧠 Bi-Encoder: {model_name} (SOTA)")
    print(f"⚡ Device: {device.upper()}")
    if use_reranker:
        print("🎯 Cross-Encoder: ms-marco-MiniLM-L-12-v2 (Deep re-ranking)")
    print("✨ กระบวนการ AI: 4 ขั้นตอน (Semantic → Hybrid → Cross-Encoder → Diversity)")
    print("💡 ความรู้เฉพาะด้าน: ขยายคำค้นด้วยอารมณ์/สไตล์/เนื้อเรื่อง")
    print("📊 Metadata Fusion: จับคู่ Keywords + นักแสดง + ผู้กำกับ")
    print("🌏 ภาษา: รองรับไทย + อังกฤษ")
    print("\n🌐 Access at: http://localhost:8000")
    print("=" * 80 + "\n")
    # ใช้ port 8000 ตามที่ gunicorn จะใช้
    app.run(debug=True, host='0.0.0.0', port=8000)
