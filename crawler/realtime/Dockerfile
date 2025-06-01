# Sử dụng image Python nhẹ
FROM python:3.11.4-slim

# Đặt thư mục làm việc
WORKDIR /app

# Copy file requirements (nếu có)
COPY requirements.txt .

# Cài đặt các thư viện từ requirements.txt
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Copy file mã nguồn của Spark Streaming job
COPY . .

# Command chạy Spark Streaming job
CMD ["python", "crawler-realtime.py"]