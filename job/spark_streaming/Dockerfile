# Sử dụng base image Bitnami/Spark
FROM bitnami/spark:3.5.0

# Cài đặt Python packages
COPY requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Chuyển sang root để cài thêm gói
USER root

# Tạo thư mục apt lists and partial, rồi update và install wget + ca-certificates
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*


RUN wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/2.8.0/kafka-clients-2.8.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-streaming_2.12/3.5.0/spark-streaming_2.12-3.5.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/scala-lang/scala-library/2.12.18/scala-library-2.12.18.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-10_2.12/3.5.0/spark-streaming-kafka-0-10_2.12-3.5.0.jar

# (Tùy chọn) Kiểm tra JAR files đã tải
RUN ls -l $SPARK_HOME/jars/ | grep -E 'spark-sql-kafka-0-10_2.12-3.5.0.jar|kafka-clients-2.8.0.jar|spark-streaming_2.12-3.5.0.jar|scala-library-2.12.18.jar|spark-streaming-kafka-0-10_2.12-3.5.0.jar'

# Đặt thư mục làm việc
WORKDIR /app

# Copy source code
COPY . .

# Chạy ứng dụng Spark trên YARN client mode
CMD ["spark-submit", \
     "--master", "yarn", \
     "--deploy-mode", "client", \
     "--jars", "/opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar,/opt/bitnami/spark/jars/kafka-clients-2.8.0.jar,/opt/bitnami/spark/jars/spark-streaming_2.12-3.5.0.jar,/opt/bitnami/spark/jars/scala-library-2.12.18.jar,/opt/bitnami/spark/jars/spark-streaming-kafka-0-10_2.12-3.5.0.jar", \
     "speed_layer.py"]
