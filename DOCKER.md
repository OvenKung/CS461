# Movie Recommendation System - Docker Guide

## 🐳 การใช้งาน Docker

### Build Docker Image
```bash
docker build -t movie-recommender .
```

### Run Container
```bash
docker run -p 5002:5002 movie-recommender
```

### Run Container (แบบ detached)
```bash
docker run -d -p 5002:5002 --name movie-app movie-recommender
```

### ดู Logs
```bash
docker logs movie-app
```

### หยุด Container
```bash
docker stop movie-app
```

### ลบ Container
```bash
docker rm movie-app
```

### เข้าถึงแอพ
เปิดเบราว์เซอร์ไปที่: http://localhost:5002

## 📦 Docker Compose (ถ้าต้องการ)

สร้างไฟล์ `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5002:5002"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

รันด้วย:
```bash
docker-compose up -d
```

## 🔧 Tips
- Image size จะประมาณ 2-3 GB เนื่องจากมี PyTorch และ sentence-transformers
- ต้องมีไฟล์ `data/movies.pkl` และ `data/movie_embeddings.npy` ในเครื่อง
- หากต้องการลดขนาด image ให้ใช้ multi-stage build
