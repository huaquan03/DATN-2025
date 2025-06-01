from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, avg, sum as sum_, when, abs
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, StringType
import clickhouse_connect
import logging
from retrying import retry

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("VN30SpeedLayer") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "true") \
    .getOrCreate()

# Khởi tạo ClickHouse client
clickhouse_client = None

# Schema
schema = StructType([
    StructField("time", TimestampType(), False),
    StructField("open", DoubleType(), False),
    StructField("high", DoubleType(), False),
    StructField("low", DoubleType(), False),
    StructField("close", DoubleType(), False),
    StructField("volume", DoubleType(), False),
    StructField("ticker", StringType(), False),
])

# Đọc Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka1:9093,kafka2:9093,kafka3:9093") \
    .option("subscribe", "GIACKREALTIME") \
    .option("startingOffsets", "latest") \
    .load()

# Parse JSON
stream_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*") \
    .filter(col("data").isNotNull()) \
    .withWatermark("time", "10 minutes")

# Tính MA1, VWAP, và các chỉ số khác
windowed_df = stream_df \
    .groupBy(
        window(col("time"), "1 minute", "1 minute").alias("window"),
        col("ticker")
    ) \
    .agg(
        avg("close").alias("ma1"),
        sum_(col("close") * col("volume")).alias("price_volume_sum"),
        sum_("volume").alias("volume_sum"),
        avg("volume").alias("avg_volume")
    ) \
    .select(
        col("ticker"),
        col("ma1"),
        col("price_volume_sum"),
        col("volume_sum"),
        col("avg_volume"),
        col("window.end").alias("time")
    ) \
    .withWatermark("time", "10 minutes") \
    .withColumn("vwap", col("price_volume_sum") / col("volume_sum"))

# Tính MA5
ma5_df = stream_df \
    .groupBy(
        window(col("time"), "5 minutes", "1 minute").alias("window"),
        col("ticker")
    ) \
    .agg(avg("close").alias("ma5")) \
    .select(col("ticker"), col("ma5"), col("window.end").alias("time"))

# Tính MA20
ma20_df = stream_df \
    .groupBy(
        window(col("time"), "20 minutes", "1 minute").alias("window"),
        col("ticker")
    ) \
    .agg(avg("close").alias("ma20")) \
    .select(col("ticker"), col("ma20"), col("window.end").alias("time"))

# Kết hợp MA5, MA20, VWAP
windowed_df = windowed_df.join(ma5_df, ["ticker", "time"], "left") \
                         .join(ma20_df, ["ticker", "time"], "left") \
                         .select(
                             col("ticker"),
                             col("ma5"),
                             col("ma20"),
                             col("vwap"),
                             col("avg_volume"),
                             col("time")
                         )

# Phát hiện bất thường (is_volume_spike)
anomaly_df = stream_df.join(
    windowed_df.select("ticker", "avg_volume", "time"),
    ["ticker", "time"],
    "left"
) \
    .withWatermark("time", "10 minutes") \
    .withColumn("is_volume_spike", when(col("volume") > 3 * col("avg_volume"), 1).otherwise(0)) \
    .select(
        col("ticker"),
        col("open"),
        col("high"),
        col("low"),
        col("close"),
        col("volume"),
        col("time"),
        col("is_volume_spike")
    )

# Kết hợp
result_df = anomaly_df.join(windowed_df, ["ticker", "time"]) \
                     .withWatermark("time", "10 minutes")

# Lưu vào ClickHouse với retry
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def write_to_clickhouse(batch_df, batch_id):
    global clickhouse_client
    try:
        if batch_df.isEmpty():
            logger.info(f"Batch {batch_id} rỗng, bỏ qua")
            return
        pdf = batch_df.toPandas()
        if not pdf.empty:
            if not clickhouse_client:
                clickhouse_client = clickhouse_connect.get_client(
                    host='clickhouse',
                    port=9000,
                    username='default',
                    password=''
                )
            clickhouse_client.insert_df(
                table='vn30_realtime',
                df=pdf,
                column_names=[
                    'ticker', 'open', 'high', 'low', 'close', 'volume',
                    'ma5', 'ma20', 'vwap', 'is_volume_spike', 'time'
                ]
            )
            logger.info(f"Batch {batch_id} appended vào ClickHouse")
    except Exception as e:
        logger.error(f"Lỗi append ClickHouse batch {batch_id}: {e}")
        raise

# Bắt đầu stream
query = result_df.writeStream \
    .outputMode("append") \
    .foreachBatch(write_to_clickhouse) \
    .option("checkpointLocation", "/mnt/checkpoints/speed_layer") \
    .trigger(processingTime="1 minute") \
    .start()

try:
    query.awaitTermination()
except Exception as e:
    logger.error(f"Lỗi trong quá trình streaming: {e}")