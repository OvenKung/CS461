# ใช้ Python 3.11 slim image เพื่อลดขนาด
FROM python:3.11-slim

# ตั้งค่า working directory
WORKDIR /app

# ติดตั้ง system dependencies ที่จำเป็น (ถ้ามี)
# RUN apt-get update && apt-get install -y ...

# คัดลอก requirements.txt
COPY requirements.txt .

# ติดตั้ง Python dependencies
# --no-cache-dir เพื่อลดขนาด image
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจคทั้งหมด
COPY . .

# Expose port ที่ gunicorn จะทำงาน
EXPOSE 8000

# ตั้งค่า environment variables
ENV PYTHONUNBUFFERED=1

# รัน Flask app ด้วย Gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
CMD gunicorn --bind 0.0.0.0:8000 app:app

