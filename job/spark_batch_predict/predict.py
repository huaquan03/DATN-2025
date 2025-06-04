import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from pyspark.sql import SparkSession
import logging
from datetime import datetime, timedelta
import tempfile
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def next_trading_day(current_date, days):
    date = pd.to_datetime(current_date)
    for _ in range(days):
        date += timedelta(days=1)
        while date.weekday() >= 5:  # Bỏ qua thứ Bảy và Chủ nhật
            date += timedelta(days=1)
    return date

def predict_future():
    # Khởi tạo SparkSession
    try:
        spark = SparkSession.builder \
            .appName("VN30FuturePrediction") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.cores", "2") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("INFO")
        logger.info("Khởi tạo SparkSession thành công")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo SparkSession: {e}")
        return None

    # Đọc dữ liệu từ HDFS
    try:
        spark_df = spark.read.parquet("hdfs://namenode:9000/data/vn30_features.parquet")
        df = spark_df.toPandas()
        logger.info(f"Đã đọc file vn30_features.parquet, số dòng: {len(df)}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu từ HDFS: {e}")
        spark.stop()
        return None

    # Danh sách features và targets
    features = ['time_idx', 'ceiling_floor', 'price_change_pct', 'historical_volatility_20',
                'volume_ma5', 'volume_ma20', 'volume_change_pct', 'sma5_pct', 'sma20_pct',
                'sma50_pct', 'rsi14', 'bb_high_pct', 'bb_low_pct', 'bb_mid_pct',
                'macd_pct', 'macd_signal_pct', 'day_of_week', 'month', 'days_since_start',
                'close_lag1_pct', 'volume_lag1', 'close_lag3_pct', 'volume_lag3',
                'close_lag5_pct', 'volume_lag5']
    targets = ['target_close_t+1', 'target_close_t+3', 'target_close_t+5']
    
    # Kiểm tra cột
    missing_cols = [col for col in features + targets + ['time', 'ticker', 'close'] if col not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu các cột: {missing_cols}")
        spark.stop()
        return None

    # Khởi tạo HDFS filesystem với Spark
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo HDFS filesystem: {e}")
        spark.stop()
        return None

    # Tải mô hình và scaler từ HDFS
    models = {}
    scalers = {}
    for ticker in df['ticker'].unique():
        for target in targets:
            hdfs_dir = f"/model/ticker={ticker}"
            model_path = f"{hdfs_dir}/model_finetune_{target}.h5"
            scaler_X_path = f"{hdfs_dir}/scaler_X_{target}.pkl"
            scaler_y_path = f"{hdfs_dir}/scaler_y_{target}.pkl"
            
            try:
                # Đọc mô hình từ HDFS vào file tạm cục bộ
                hdfs_model_path = spark._jvm.org.apache.hadoop.fs.Path(model_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file:
                    with fs.open(hdfs_model_path) as f:
                        temp_model_file.write(f.readAllBytes())
                    model = load_model(temp_model_file.name)
                    os.remove(temp_model_file.name)
                
                # Đọc scaler_X từ HDFS vào file tạm cục bộ
                hdfs_scaler_X_path = spark._jvm.org.apache.hadoop.fs.Path(scaler_X_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_scaler_X_file:
                    with fs.open(hdfs_scaler_X_path) as f:
                        temp_scaler_X_file.write(f.readAllBytes())
                    scaler_X = pd.read_pickle(temp_scaler_X_file.name)
                    os.remove(temp_scaler_X_file.name)
                
                # Đọc scaler_y từ HDFS vào file tạm cục bộ
                hdfs_scaler_y_path = spark._jvm.org.apache.hadoop.fs.Path(scaler_y_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_scaler_y_file:
                    with fs.open(hdfs_scaler_y_path) as f:
                        temp_scaler_y_file.write(f.readAllBytes())
                    scaler_y = pd.read_pickle(temp_scaler_y_file.name)
                    os.remove(temp_scaler_y_file.name)
                
                models[(ticker, target)] = model
                scalers[(ticker, target)] = (scaler_X, scaler_y)
                logger.info(f"Đã tải mô hình và scaler cho {ticker} - {target}")
            except Exception as e:
                logger.warning(f"Lỗi khi tải mô hình/scaler cho {ticker} - {target}: {e}")
                continue
    
    # Dự đoán cho từng mã
    predictions = []
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].tail(30)[['time', 'ticker', 'close'] + features]
        if len(ticker_data) < 30:
            logger.warning(f"Không đủ dữ liệu cho {ticker} (chỉ có {len(ticker_data)} dòng)")
            continue
        pred_row = ticker_data.tail(1)[['time', 'ticker', 'close']].copy()
        for target, days in zip(targets, [1, 3, 5]):
            if (ticker, target) not in models:
                pred_row[f'pred_{target}'] = np.nan
                pred_row[f'pred_time_t+{days}'] = np.nan
                continue
            model = models[(ticker, target)]
            scaler_X, scaler_y = scalers[(ticker, target)]
            X_latest = ticker_data[features]
            X_latest_scaled = scaler_X.transform(X_latest)
            X_latest_reshaped = np.array([X_latest_scaled[-30:]])
            try:
                pred_scaled = model.predict(X_latest_reshaped)
                pred_row[f'pred_{target}'] = scaler_y.inverse_transform(pred_scaled).flatten()[0]
                pred_row[f'pred_time_t+{days}'] = next_trading_day(pred_row['time'].iloc[0], days)
            except Exception as e:
                logger.error(f"Lỗi khi dự đoán cho {ticker} - {target}: {e}")
                pred_row[f'pred_{target}'] = np.nan
                pred_row[f'pred_time_t+{days}'] = np.nan
        predictions.append(pred_row)
    
    if not predictions:
        logger.error("Không có dự đoán nào được tạo ra")
        spark.stop()
        return None
    
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df = predictions_df[['time', 'ticker', 'close',
                                    'pred_time_t+1', 'pred_target_close_t+1',
                                    'pred_time_t+3', 'pred_target_close_t+3',
                                    'pred_time_t+5', 'pred_target_close_t+5']]
    
    # Ghi kết quả dự đoán vào HDFS bằng fastparquet
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_parquet_file:
            predictions_df.to_parquet(temp_parquet_file.name, engine='fastparquet', compression='snappy')
            hdfs_path = spark._jvm.org.apache.hadoop.fs.Path("hdfs://namenode:9000/model/predictions.parquet")
            with open(temp_parquet_file.name, 'rb') as local_file:
                with fs.create(hdfs_path) as hdfs_file:
                    hdfs_file.write(local_file.read())
            os.remove(temp_parquet_file.name)
        logger.info("Kết quả dự đoán đã được ghi vào HDFS tại: hdfs://namenode:9000/model/predictions.parquet")
    except Exception as e:
        logger.error(f"Lỗi khi lưu kết quả dự đoán vào HDFS: {e}")
        spark.stop()
        return None

    spark.stop()
    return predictions_df

if __name__ == "__main__":
    predictions_df = predict_future()