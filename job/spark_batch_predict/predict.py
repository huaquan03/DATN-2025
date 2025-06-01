import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from pyspark.sql import SparkSession
import joblib
import logging
import os
from datetime import datetime, timedelta

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def next_trading_day(current_date, days):
    date = pd.to_datetime(current_date)
    for _ in range(days):
        date += timedelta(days=1)
        while date.weekday() >= 5:
            date += timedelta(days=1)
    return date

def predict_future():
    # Khởi tạo SparkSession
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
    
    # Đọc dữ liệu từ HDFS
    try:
        spark_df = spark.read.parquet("hdfs://namenode:9000/data/vn30_features.parquet")
        df = spark_df.toPandas()
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
    missing_cols = [col for col in features + ['time', 'ticker', 'close'] if col not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu các cột: {missing_cols}")
        spark.stop()
        return None
    
    # Tải mô hình và scaler từ HDFS
    models = {}
    scalers = {}
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    for ticker in df['ticker'].unique():
        for target in targets:
            hdfs_dir = f"/model/ticker={ticker}"
            model_path = f"{hdfs_dir}/model_finetune_{target}.h5"
            scaler_X_path = f"{hdfs_dir}/scaler_X_{target}.pkl"
            scaler_y_path = f"{hdfs_dir}/scaler_y_{target}.pkl"
            
            try:
                # Tải file từ HDFS về cục bộ
                for path, local_file in [(model_path, f"temp_model_{ticker}_{target}.h5"),
                                       (scaler_X_path, f"temp_scaler_X_{ticker}_{target}.pkl"),
                                       (scaler_y_path, f"temp_scaler_y_{ticker}_{target}.pkl")]:
                    if fs.exists(spark._jvm.org.apache.hadoop.fs.Path(path)):
                        fs.copyToLocalFile(False, spark._jvm.org.apache.hadoop.fs.Path(path),
                                         spark._jvm.org.apache.hadoop.fs.Path(local_file))
                    else:
                        logger.warning(f"Không tìm thấy {path} cho {ticker} - {target}")
                        continue
                
                model = load_model(f"temp_model_{ticker}_{target}.h5")
                scaler_X = joblib.load(f"temp_scaler_X_{ticker}_{target}.pkl")
                scaler_y = joblib.load(f"temp_scaler_y_{ticker}_{target}.pkl")
                
                models[(ticker, target)] = model
                scalers[(ticker, target)] = (scaler_X, scaler_y)
                logger.info(f"Đã tải mô hình và scaler cho {ticker} - {target}")
                
                # Dọn dẹp file tạm
                for local_file in [f"temp_model_{ticker}_{target}.h5",
                                 f"temp_scaler_X_{ticker}_{target}.pkl",
                                 f"temp_scaler_y_{ticker}_{target}.pkl"]:
                    if os.path.exists(local_file):
                        os.remove(local_file)
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình/scaler cho {ticker} - {target}: {e}")
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
    
    # Ghi kết quả dự đoán vào HDFS
    pred_spark_df = spark.createDataFrame(predictions_df)
    pred_spark_df.write \
        .partitionBy("ticker") \
        .mode("overwrite") \
        .parquet("hdfs://namenode:9000/model/predictions.parquet")
    logger.info("Kết quả dự đoán đã được ghi vào HDFS tại: hdfs://namenode:9000/model/predictions.parquet")
    
    # Lưu cục bộ để kiểm tra
    predictions_df.to_csv('predictions.csv', index=False)
    logger.info("Kết quả dự đoán đã được lưu cục bộ tại: predictions.csv")
    
    spark.stop()
    return predictions_df

if __name__ == "__main__":
    predictions_df = predict_future()