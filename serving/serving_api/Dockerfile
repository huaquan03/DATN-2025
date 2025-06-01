# Sử dụng image Python nhẹ
FROM python:3.8-slim

# Đặt thư mục làm việc
WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Copy mã nguồn của API
COPY . .

# Command chạy ứng dụng
CMD ["python", "app.py"]
