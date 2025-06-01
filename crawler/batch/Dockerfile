FROM bitnami/spark:3.5.0
USER root

# Cài thêm Python và pip
RUN install_packages python3 python3-pip

# Cài các thư viện Python yêu cầu
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Thiết lập HADOOP_CONF_DIR
ENV HADOOP_CONF_DIR=/opt/hadoop/conf

# Copy mã nguồn
COPY . /app
WORKDIR /app

# Chạy spark-submit với cấu hình Python
CMD ["spark-submit", \
     "--master", "yarn", \
     "--deploy-mode", "client", \
     "--name", "crawler-batch-job", \
     "--num-executors", "2", \
     "--executor-cores", "1", \
     "--executor-memory", "2500m", \
     "--conf", "spark.executor.memoryOverhead=384m", \
     "--driver-memory", "1024m", \
     "--conf", "spark.driver.memoryOverhead=512m", \
     "--conf", "spark.driver.bindAddress=0.0.0.0", \
     "--conf", "spark.driver.host=crawler-batch", \
     "--conf", "spark.driver.port=47000", \
     "--conf", "spark.blockManager.port=47001", \
     "--conf", "spark.yarn.appMasterEnv.PYSPARK_PYTHON=/usr/bin/python3", \
     "--conf", "spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3", \
     "/app/crawler-batch.py"]