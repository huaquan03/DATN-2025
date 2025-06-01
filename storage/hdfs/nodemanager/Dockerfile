# ---------------------------------------------------------------
# 1. Base image: Ubuntu 20.04 LTS
FROM ubuntu:20.04

# 2. Đặt môi trường để apt không tương tác
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------
# 3. Cài OpenJDK 8 (Java 8) cho Hadoop 2.7.4
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         openjdk-8-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# 4. Thêm PPA deadsnakes để cài Python 3.11
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# 5. Cài Python 3.11 và distutils (cần để bootstrap pip)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         python3.11 \
         python3.11-distutils \
         wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# 6. Symlink python3 → python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# ---------------------------------------------------------------
# 7. Bootstrapping pip cho Python 3.11
#    - Tải get-pip.py chính thức và chạy bằng python3.11 để cài pip
RUN wget --no-verbose https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

# ---------------------------------------------------------------
# 8. Cài PySpark 3.5.0 bằng pip3 (pip3 bây giờ thuộc Python 3.11)
RUN python3.11 -m pip install --no-cache-dir pyspark==3.5.0
# ==== Bắt đầu phần thêm TensorFlow ====
# 7. Cài TensorFlow CPU (phiên bản stable)
#    - Bạn có thể đổi "tensorflow" thành "tensorflow==2.x.y" theo version mong muốn.
RUN python3.11 -m pip install --no-cache-dir tensorflow

# ==== Kết thúc phần thêm TensorFlow ====
# ---------------------------------------------------------------
# 9. Thiết lập biến môi trường Hadoop & PySpark
ENV HADOOP_VERSION=2.7.4
ENV HADOOP_HOME=/opt/hadoop-2.7.4
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
# Spark worker và driver sẽ dùng python3 (đã symlink → python3.11)
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ENV PATH="$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin"

# ---------------------------------------------------------------
# 10. Download và giải nén Hadoop 2.7.4
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         wget tar \
    && rm -rf /var/lib/apt/lists/* \
    && wget --no-verbose -O /tmp/hadoop-2.7.4.tar.gz \
         https://archive.apache.org/dist/hadoop/common/hadoop-2.7.4/hadoop-2.7.4.tar.gz \
    && tar -xzvf /tmp/hadoop-2.7.4.tar.gz -C /opt/ \
    && rm /tmp/hadoop-2.7.4.tar.gz


# ---------------------------------------------------------------
# 13. ENTRYPOINT: Khởi NodeManager và giữ container chạy
ENTRYPOINT ["bash", "-c", "yarn nodemanager -daemon && tail -f /dev/null"]
