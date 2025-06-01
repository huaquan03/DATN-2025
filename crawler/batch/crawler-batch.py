import pandas as pd
import numpy as np
from vnstock import Vnstock
from pyspark.sql import SparkSession
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from forhdfs import feature_engineering_for_hdfs, push_to_hdfs
from forclickhouse import feature_engineering_for_clickhouse, push_to_clickhouse
# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hàm lấy dữ liệu cho từng mã
def fetch_stock_data(symbol, start, end):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=start, end=end, interval='1D')
        if not df.empty:
            df['ticker'] = symbol
            df = df.rename(columns={'date': 'time'})
            return df
        return None
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu cho {symbol}: {e}")
        return None

def main():
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("VN30FeatureEngineeringToHDFS") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    
    # Kiểm tra HDFS
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        fs.exists(spark._jvm.org.apache.hadoop.fs.Path("/"))
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra HDFS: {e}")
        spark.stop()
        return
    
    # Khởi tạo đối tượng stock
    stock = Vnstock().stock(symbol='ACB', source='VCI')
    
    # Lấy danh sách mã VN30
    df_symbols = stock.listing.symbols_by_group('VN30')
    
    # Thu thập dữ liệu song song
    start = '2005-01-01'
    end = datetime.now().strftime('%Y-%m-%d')
    all_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(lambda s: fetch_stock_data(s, start, end), df_symbols)
        all_data = [r for r in results if r is not None]
    
    # Gộp dữ liệu
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(subset=['close', 'volume'])
        combined_df = combined_df[(combined_df['close'] > 0) & (combined_df['volume'] > 0)]
        
        if combined_df.empty:
            logger.error("Dữ liệu gốc rỗng, dừng xử lý.")
            spark.stop()
            return
        
        # Lưu dữ liệu gốc vào HDFS
        raw_spark_df = spark.createDataFrame(combined_df)
        raw_spark_df.write \
            .partitionBy("ticker") \
            .mode("overwrite") \
            .parquet("hdfs://namenode:9000/data/vn30_raw.parquet")
        logger.info("Dữ liệu gốc đã được ghi vào HDFS tại: hdfs://namenode:9000/data/vn30_raw.parquet")
        
        
        # Feature engineering cho HDFS và ClickHouse
        tickers = combined_df['ticker'].unique()
        processed_dfs_hdfs = []
        processed_dfs_clickhouse = []
        
        for ticker in tickers:
            ticker_df = combined_df[combined_df['ticker'] == ticker].copy()
            if len(ticker_df) > 50:
                # Xử lý cho HDFS
                processed_df_hdfs = feature_engineering_for_hdfs(ticker_df)
                if processed_df_hdfs is not None and not processed_df_hdfs.empty:
                    processed_dfs_hdfs.append(processed_df_hdfs)
                    logger.info(f"Feature engineering cho HDFS của {ticker} hoàn tất.")
                else:
                    logger.warning(f"Feature engineering cho HDFS của {ticker} thất bại.")
                
                # Xử lý cho ClickHouse
                processed_df_clickhouse = feature_engineering_for_clickhouse(ticker_df)
                if processed_df_clickhouse is not None and not processed_df_clickhouse.empty:
                    processed_dfs_clickhouse.append(processed_df_clickhouse)
                    logger.info(f"Feature engineering cho ClickHouse của {ticker} hoàn tất.")
                else:
                    logger.warning(f"Feature engineering cho ClickHouse của {ticker} thất bại.")
            else:
                logger.warning(f"Không đủ dữ liệu cho {ticker} (dưới 50 dòng).")
        
        # Đẩy dữ liệu vào HDFS
        if processed_dfs_hdfs:
            final_df_hdfs = pd.concat(processed_dfs_hdfs, ignore_index=True)
            success_hdfs = push_to_hdfs(spark, final_df_hdfs)
            if success_hdfs:
                logger.info("Đẩy dữ liệu vào HDFS thành công")
            else:
                logger.error("Đẩy dữ liệu vào HDFS thất bại")
        else:
            logger.error("Không có mã cổ phiếu nào đủ dữ liệu cho HDFS.")
        
        # Đẩy dữ liệu vào ClickHouse
        if processed_dfs_clickhouse:
            final_df_clickhouse = pd.concat(processed_dfs_clickhouse, ignore_index=True)
            success_clickhouse = push_to_clickhouse(final_df_clickhouse)
            if success_clickhouse:
                logger.info("Đẩy dữ liệu vào ClickHouse thành công")
            else:
                logger.error("Đẩy dữ liệu vào ClickHouse thất bại")
        else:
            logger.error("Không có mã cổ phiếu nào đủ dữ liệu cho ClickHouse.")
    else:
        logger.error("Không thu thập được dữ liệu từ bất kỳ mã cổ phiếu nào.")
    
    spark.stop()

if __name__ == "__main__":
    main()