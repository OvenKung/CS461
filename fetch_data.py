import requests
import pandas as pd
import time
import os
import pickle
from tmdb_client import TMDBClient
from tqdm import tqdm

def fetch_fresh_data():
    client = TMDBClient()
    if not client.api_key:
        print("❌ No API Key found. Please set TMDB_API_KEY environment variable.")
        return

    print(f"🚀 Starting data fetch with API Key: {client.api_key[:5]}...")
    
    movies_dict = {} # Use dict to deduplicate by ID
    
    # Endpoints to fetch
    endpoints = [
        ("movie/popular", 100),      # 2000 movies
        ("movie/top_rated", 50),     # 1000 movies
        ("movie/now_playing", 20),   # 400 movies
        ("movie/upcoming", 20)       # 400 movies
    ]
    
    total_movies_fetched = 0
    
    for endpoint, max_pages in endpoints:
        print(f"📥 Fetching {endpoint}...")
        for page in tqdm(range(1, max_pages + 1)):
            try:
                url = f"{client.base_url}/{endpoint}"
                params = {
                    "api_key": client.api_key,
                    "language": "en-US",
                    "page": page
                }
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    for movie in results:
                        movie_id = movie['id']
                        if movie_id not in movies_dict:
                            movies_dict[movie_id] = movie
                elif response.status_code == 429:
                    print("⚠️ Rate limit hit. Sleeping for 5s...")
                    time.sleep(5)
                    continue
                else:
                    print(f"⚠️ Error fetching {endpoint} page {page}: {response.status_code}")
                
                # Respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)

    print(f"✅ Found {len(movies_dict)} unique movies. Fetching details...")
    
    # Fetch details for each movie (Keywords, Credits)
    detailed_movies = []
    movie_ids = list(movies_dict.keys())
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def fetch_details(movie_id):
        try:
            url = f"{client.base_url}/movie/{movie_id}"
            params = {
                "api_key": client.api_key,
                "language": "en-US",
                "append_to_response": "keywords,credits,release_dates"
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract certification
                certification = ''
                release_dates = data.get('release_dates', {}).get('results', [])
                for country in release_dates:
                    if country['iso_3166_1'] == 'US':
                        for release in country['release_dates']:
                            if release.get('certification'):
                                certification = release['certification']
                                break
                
                return {
                    'id': data['id'],
                    'title': data['title'],
                    'overview': data['overview'],
                    'genres': data['genres'],
                    'keywords': data.get('keywords', {}).get('keywords', []),
                    'cast': data.get('credits', {}).get('cast', []),
                    'crew': data.get('credits', {}).get('crew', []),
                    'production_companies': data.get('production_companies', []),
                    'vote_average': data['vote_average'],
                    'vote_count': data['vote_count'],
                    'release_date': data['release_date'],
                    'popularity': data['popularity'],
                    'certification': certification,
                    'poster_path': data['poster_path'],
                    'backdrop_path': data['backdrop_path']
                }
            elif response.status_code == 429:
                time.sleep(1) # Backoff
                return None # Skip or retry (simplified to skip for speed)
        except Exception as e:
            print(f"❌ Error fetching details for {movie_id}: {e}")
            return None
        return None

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_details, mid) for mid in movie_ids]
        for future in tqdm(as_completed(futures), total=len(movie_ids), desc="Fetching details"):
            result = future.result()
            if result:
                detailed_movies.append(result)

    # Save to pickle
    os.makedirs('data', exist_ok=True)
    with open('data/movies_fresh.pkl', 'wb') as f:
        pickle.dump(detailed_movies, f)
        
    print(f"🎉 Successfully saved {len(detailed_movies)} movies to data/movies_fresh.pkl")

if __name__ == "__main__":
    fetch_fresh_data()
