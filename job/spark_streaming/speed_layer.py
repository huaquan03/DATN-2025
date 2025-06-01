from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import logging
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_streaming():
    # Lấy biến môi trường
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9093")

    # Khởi tạo SparkSession cho YARN client mode
    spark = SparkSession.builder \
        .appName("VN30Streaming") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.instances", "4") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.streaming.backpressure.enabled", "true") \
        .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    logger.info("Khởi tạo SparkSession cho YARN client mode thành công")

    # Định nghĩa schema cho dữ liệu Kafka
    schema = StructType([
        StructField("ticker", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("timestamp", StringType(), True)
    ])

    try:
        # Đọc stream từ Kafka
        df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("subscribe", "vn30-topic") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse dữ liệu JSON
        parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
        
        # Xử lý dữ liệu
        processed_df = parsed_df.groupBy("ticker").agg({"price": "avg", "volume": "sum"})
        
        # Ghi stream ra console
        query = processed_df.writeStream \
            .outputMode("update") \
            .format("console") \
            .trigger(processingTime="10 seconds") \
            .start()
        
        query.awaitTermination()
    except Exception as e:
        logger.error(f"Lỗi trong streaming: {e}")
    finally:
        spark.stop()
        logger.info("Đã dừng SparkSession")

if __name__ == "__main__":
    run_streaming()