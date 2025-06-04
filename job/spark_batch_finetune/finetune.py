import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lstm(X, y, base_model, timesteps=30, epochs=20, batch_size=32, learning_rate=0.00005, fine_tune=True):
    try:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
        
        X_reshaped = []
        y_reshaped = []
        for i in range(timesteps, len(X_scaled)):
            X_reshaped.append(X_scaled[i-timesteps:i])
            y_reshaped.append(y_scaled[i])
        X_reshaped = np.array(X_reshaped)
        y_reshaped = np.array(y_reshaped)
        
        if X_reshaped.shape[0] == 0:
            raise ValueError("Dữ liệu không đủ để tạo chuỗi thời gian")
        
        model = clone_model(base_model)
        model.set_weights(base_model.get_weights())
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(X_reshaped, y_reshaped, epochs=epochs, batch_size=batch_size,
                 callbacks=[early_stopping], verbose=1)
        
        return model, scaler_X, scaler_y
    except Exception as e:
        logger.error(f"Lỗi trong train_lstm: {e}")
        raise

def split_finetune():
    # Đợi Namenode sẵn sàng
    logger.info("Đợi 10 giây để Namenode sẵn sàng...")
    time.sleep(10)
    
    # Khởi tạo SparkSession
    try:
        spark = SparkSession.builder \
            .appName("SplitVN30Finetune") \
            .master("local[*]") \
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
            .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.cores", "6") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("INFO")
        logger.info("Khởi tạo SparkSession thành công")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo SparkSession: {e}")
        return

    # Đọc mô hình chung từ HDFS trực tiếp vào bộ nhớ
    try:
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        hdfs_path = spark._jvm.org.apache.hadoop.fs.Path("/model/temp_common_model.h5")
        with fs.open(hdfs_path) as f:
            model_content = f.readAllBytes()
        
        with BytesIO(model_content) as model_buffer:
            base_model = load_model(model_buffer)
        logger.info("Đã tải mô hình temp_common_model.h5 từ HDFS vào bộ nhớ")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình chung từ HDFS: {e}")
        spark.stop()
        return

    # Đọc dữ liệu từ HDFS
    try:
        spark_df = spark.read.parquet("hdfs://namenode:9000/data/vn30_features.parquet")
        df = spark_df.toPandas()
        logger.info(f"Đã đọc file vn30_features.parquet, số dòng: {len(df)}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu từ HDFS: {e}")
        spark.stop()
        return

    # Định nghĩa đặc trưng và mục tiêu
    features = ['time_idx', 'ceiling_floor', 'price_change_pct', 'historical_volatility_20',
                'volume_ma5', 'volume_ma20', 'volume_change_pct', 'sma5_pct', 'sma20_pct',
                'sma50_pct', 'rsi14', 'bb_high_pct', 'bb_low_pct', 'bb_mid_pct',
                'macd_pct', 'macd_signal_pct', 'day_of_week', 'month', 'days_since_start',
                'close_lag1_pct', 'volume_lag1', 'close_lag3_pct', 'volume_lag3',
                'close_lag5_pct', 'volume_lag5']
    targets = ['target_close_t+1', 'target_close_t+3', 'target_close_t+5']

    # Kiểm tra cột
    missing_cols = [col for col in features + targets if col not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu các cột: {missing_cols}")
        spark.stop()
        return

    # Fine-tune mô hình cho từng mã
    def fine_tune_ticker(ticker):
        ticker_df = df[df['ticker'] == ticker]
        hdfs_dir = f"/model/ticker={ticker}"
        
        for target in targets:
            X = ticker_df[features]
            y = ticker_df[target]
            if len(X) < 100:
                logger.warning(f"Bỏ qua {ticker} - {target}: dữ liệu ngắn ({len(X)} dòng)")
                continue
            try:
                model, scaler_X, scaler_y = train_lstm(X, y, base_model, timesteps=30, epochs=20, fine_tune=True)
                
                # Lưu mô hình và scaler vào file tạm cục bộ
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_model_file, \
                     tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_scaler_X_file, \
                     tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_scaler_y_file:
                    
                    # Lưu mô hình và scaler
                    model.save(temp_model_file.name)
                    pd.to_pickle(scaler_X, temp_scaler_X_file.name)
                    pd.to_pickle(scaler_y, temp_scaler_y_file.name)
                    
                    # Lưu lên HDFS
                    for file_name, temp_file in [
                        (f"{hdfs_dir}/model_finetune_{target}.h5", temp_model_file),
                        (f"{hdfs_dir}/scaler_X_{target}.pkl", temp_scaler_X_file),
                        (f"{hdfs_dir}/scaler_y_{target}.pkl", temp_scaler_y_file)
                    ]:
                        hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(file_name)
                        with open(temp_file.name, 'rb') as local_file:
                            with fs.create(hdfs_path) as hdfs_file:
                                hdfs_file.write(local_file.read())
                        os.remove(temp_file.name)
                
                logger.info(f"Fine-tuning cho {ticker} - {target} hoàn tất, lưu vào {hdfs_dir}")
            except Exception as e:
                logger.error(f"Lỗi khi fine-tune cho {ticker} - {target}: {e}")
                continue

    try:
        tickers = df['ticker'].unique()
        logger.info(f"Số mã cổ phiếu: {len(tickers)}")
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(fine_tune_ticker, tickers)
    except Exception as e:
        logger.error(f"Lỗi khi fine-tune: {e}")
        spark.stop()
        return

    spark.stop()
    logger.info("Hoàn tất fine-tune mô hình!")

if __name__ == "__main__":
    split_finetune()