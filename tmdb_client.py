import requests
import os
from functools import lru_cache

class TMDBClient:
    def __init__(self, api_key=None):
        # Use provided key or fallback to hardcoded key from user
        self.api_key = api_key or os.environ.get('TMDB_API_KEY') or '553122491c31dcbf2b88442bf68658f2'
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
        self.backdrop_base_url = "https://image.tmdb.org/t/p/original"
        
    def get_headers(self):
        if not self.api_key:
            return {}
        return {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }

    def search_movie(self, title, year=None):
        if not self.api_key:
            return None
            
        params = {
            "query": title,
            "language": "en-US",
            "page": 1
        }
        if year:
            params["primary_release_year"] = year
            
        try:
            # ใช้ API Key แบบ query param ถ้าไม่ได้ใช้ Bearer token
            # แต่เพื่อความปลอดภัยรองรับทั้งคู่ (ในที่นี้ใช้ Bearer เป็นหลักถ้า key ยาว, หรือ query param ถ้าสั้น)
            # เพื่อความง่าย ใช้ query param สำหรับ api_key ธรรมดา
            if len(self.api_key) < 50: # สันนิษฐานว่าเป็น API Key ธรรมดา
                params["api_key"] = self.api_key
                headers = {}
            else: # Read Access Token
                headers = self.get_headers()
                
            response = requests.get(f"{self.base_url}/search/movie", params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    return results[0] # Return best match
            return None
        except Exception as e:
            print(f"⚠️ TMDB Search Error: {e}")
            return None

    def get_movie_details(self, tmdb_id):
        if not self.api_key or not tmdb_id:
            return None
            
        try:
            params = {
                "append_to_response": "videos,watch/providers,credits",
                "language": "en-US"
            }
            
            if len(self.api_key) < 50:
                params["api_key"] = self.api_key
                headers = {}
            else:
                headers = self.get_headers()

            response = requests.get(f"{self.base_url}/movie/{tmdb_id}", params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"⚠️ TMDB Details Error: {e}")
            return None

    def enrich_movie_data(self, title, year=None):
        """ดึงข้อมูลรูปภาพและรายละเอียดเพิ่มเติม"""
        if not self.api_key:
            return {}
            
        # 1. Search for the movie
        search_result = self.search_movie(title, year)
        if not search_result:
            return {}
            
        tmdb_id = search_result['id']
        
        # 2. Get details
        details = self.get_movie_details(tmdb_id)
        if not details:
            return {}
            
        # 3. Extract useful info
        poster_path = details.get('poster_path')
        backdrop_path = details.get('backdrop_path')
        
        # Trailer
        videos = details.get('videos', {}).get('results', [])
        trailer = next((v for v in videos if v['type'] == 'Trailer' and v['site'] == 'YouTube'), None)
        
        # Streaming (TH or US)
        providers = details.get('watch/providers', {}).get('results', {})
        th_providers = providers.get('TH', {}).get('flatrate', [])
        us_providers = providers.get('US', {}).get('flatrate', [])
        streaming = th_providers if th_providers else us_providers
        
        return {
            'poster_url': f"{self.image_base_url}{poster_path}" if poster_path else None,
            'backdrop_url': f"{self.backdrop_base_url}{backdrop_path}" if backdrop_path else None,
            'tmdb_rating': details.get('vote_average'),
            'tmdb_vote_count': details.get('vote_count'),
            'trailer_id': trailer['key'] if trailer else None,
            'streaming': [{'name': p['provider_name'], 'logo': f"{self.image_base_url}{p['logo_path']}"} for p in streaming[:3]]
        }
