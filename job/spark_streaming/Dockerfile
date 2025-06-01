# Sử dụng image cơ bản của Python
FROM python:3.11

# Cài đặt các công cụ cần thiết và Java
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        wget \
        procps \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Đặt biến môi trường cho Java
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Cài đặt Apache Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt/ && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

# Đặt biến môi trường cho Spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Tải Kafka connector, kafka-clients, spark-streaming, và các phụ thuộc bổ sung
RUN wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/2.8.0/kafka-clients-2.8.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-streaming_2.12/3.5.0/spark-streaming_2.12-3.5.0.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/scala-lang/scala-library/2.12.18/scala-library-2.12.18.jar && \
    wget -q -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/spark/spark-streaming-kafka-0-10_2.12/3.5.0/spark-streaming-kafka-0-10_2.12-3.5.0.jar

# Kiểm tra tính toàn vẹn của JAR files
RUN ls -l $SPARK_HOME/jars/ | grep -E 'spark-sql-kafka-0-10_2.12-3.5.0.jar|kafka-clients-2.8.0.jar|spark-streaming_2.12-3.5.0.jar|scala-library-2.12.18.jar|spark-streaming-kafka-0-10_2.12-3.5.0.jar'

# Đặt thư mục làm việc
WORKDIR /app

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Chạy ứng dụng Spark với Kafka connector
CMD ["spark-submit", "--master", "local[*]", "--jars", "/opt/spark/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar,/opt/spark/jars/kafka-clients-2.8.0.jar,/opt/spark/jars/spark-streaming_2.12-3.5.0.jar,/opt/spark/jars/scala-library-2.12.18.jar,/opt/spark/jars/spark-streaming-kafka-0-10_2.12-3.5.0.jar", "speed_layer.py"]