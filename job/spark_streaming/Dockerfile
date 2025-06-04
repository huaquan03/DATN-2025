FROM bitnami/spark:3.5.0

# Cài đặt Python packages từ requirements.txt
COPY requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Chuyển sang root để cài thêm gói hệ thống
USER root

# Cài đặt công cụ wget và chứng chỉ SSL
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Tải các JAR cần thiết cho Spark Kafka Streaming
ENV JARS_DIR=$SPARK_HOME/jars

RUN wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-10_2.12/3.5.0/spark-streaming-kafka-0-10_2.12-3.5.0.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/spark/spark-streaming_2.12/3.5.0/spark-streaming_2.12-3.5.0.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.0/spark-token-provider-kafka-0-10_2.12-3.5.0.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.0/kafka-clients-3.5.0.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/scala-lang/scala-library/2.12.18/scala-library-2.12.18.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.36/slf4j-api-1.7.36.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -q -P $JARS_DIR https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar

# Xác minh đã tải JARs
RUN ls -lh $JARS_DIR | grep -E 'kafka|slf4j|log4j|commons-pool2'

# Đặt thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn vào image
COPY . .

# Câu lệnh chạy Spark app
CMD ["spark-submit", \
     "--master", "yarn", \
     "--deploy-mode", "client", \
     "--jars", "/opt/bitnami/spark/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar,/opt/bitnami/spark/jars/kafka-clients-3.5.0.jar,/opt/bitnami/spark/jars/spark-streaming_2.12-3.5.0.jar,/opt/bitnami/spark/jars/scala-library-2.12.18.jar,/opt/bitnami/spark/jars/spark-streaming-kafka-0-10_2.12-3.5.0.jar,/opt/bitnami/spark/jars/spark-token-provider-kafka-0-10_2.12-3.5.0.jar,/opt/bitnami/spark/jars/slf4j-api-1.7.36.jar,/opt/bitnami/spark/jars/log4j-api-2.17.1.jar,/opt/bitnami/spark/jars/log4j-core-2.17.1.jar,/opt/bitnami/spark/jars/commons-pool2-2.11.1.jar", \
     "speed_layer.py"]