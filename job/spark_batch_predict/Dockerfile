# Sử dụng image Python nhẹ
FROM python:3.11

RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường JAVA_HOME cho OpenJDK 17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64


# Đặt thư mục làm việc
WORKDIR /app

# Copy file requirements (nếu có)
COPY requirements.txt .

# Cài đặt các thư viện từ requirements.txt
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt


# Copy file mã nguồn của Spark Streaming job
COPY predict.py .

# Command chạy Spark Streaming job
CMD ["python", "predict.py"]
