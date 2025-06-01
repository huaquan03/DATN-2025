import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import logging
import time
from io import BytesIO

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lstm(X, y, timesteps=30, epochs=50, batch_size=32, learning_rate=0.0001):
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
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(timesteps, X.shape[1])),
            Dropout(0.4),
            LSTM(64, return_sequences=True),
            Dropout(0.4),
            LSTM(32),
            Dropout(0.4),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        tscv = TimeSeriesSplit(n_splits=5)
        rmses = []
        for train_idx, val_idx in tscv.split(X_reshaped):
            X_train, X_val = X_reshaped[train_idx], X_reshaped[val_idx]
            y_train, y_val = y_reshaped[train_idx], y_reshaped[val_idx]
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_val), scaler_y.inverse_transform(y_pred)))
            rmses.append(rmse)
        logger.info(f"RMSE trung bình cross-validation: {np.mean(rmses)}")
        
        model.fit(X_reshaped, y_reshaped, epochs=epochs, batch_size=batch_size,
                 callbacks=[early_stopping], verbose=1)
        
        return model, scaler_X, scaler_y
    except Exception as e:
        logger.error(f"Lỗi trong train_lstm: {e}")
        raise

def split_train_total():
    # Đợi Namenode sẵn sàng
    logger.info("Đợi 10 giây để Namenode sẵn sàng...")
    time.sleep(10)
    
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("SplitVN30TrainTotal") \
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

    # Đọc dữ liệu từ HDFS
    try:
        spark_df = spark.read.parquet("hdfs://namenode:9000/data/vn30_features.parquet")
        df = spark_df.toPandas()
        logger.info(f"Đã đọc file vn30_features.parquet, số dòng: {len(df)}")
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu từ HDFS: {e}")
        spark.stop()
        return

    # Kiểm tra dữ liệu
    logger.info("Số dòng mỗi mã cổ phiếu:")
    logger.info(df.groupby('ticker').size())
    
    # Định nghĩa đặc trưng và mục tiêu
    features = ['time_idx', 'ceiling_floor', 'price_change_pct', 'historical_volatility_20',
                'volume_ma5', 'volume_ma20', 'volume_change_pct', 'sma5_pct', 'sma20_pct',
                'sma50_pct', 'rsi14', 'bb_high_pct', 'bb_low_pct', 'bb_mid_pct',
                'macd_pct', 'macd_signal_pct', 'day_of_week', 'month', 'days_since_start',
                'close_lag1_pct', 'volume_lag1', 'close_lag3_pct', 'volume_lag3',
                'close_lag5_pct', 'volume_lag5']
    target = 'target_close_t+1'
    
    # Kiểm tra cột
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        logger.error(f"Thiếu các cột: {missing_cols}")
        spark.stop()
        return

    # Huấn luyện mô hình chung
    try:
        X = df[features]
        y = df[target]
        if len(X) < 100:
            logger.error(f"Dữ liệu quá ngắn ({len(X)} dòng)")
            spark.stop()
            return
        model_common, scaler_X_common, scaler_y_common = train_lstm(X, y, timesteps=30, epochs=50)
        
        # Lưu mô hình và scaler vào bộ nhớ tạm
        with BytesIO() as model_buffer, BytesIO() as scaler_X_buffer, BytesIO() as scaler_y_buffer:
            model_common.save(model_buffer)
            pd.to_pickle(scaler_X_common, scaler_X_buffer)
            pd.to_pickle(scaler_y_common, scaler_y_buffer)
            
            # Lưu trực tiếp lên HDFS bằng Spark Hadoop FileSystem API
            fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
            for file_name, buffer in [
                ("temp_common_model.h5", model_buffer),
                ("temp_scaler_X_common.pkl", scaler_X_buffer),
                ("temp_scaler_y_common.pkl", scaler_y_buffer)
            ]:
                hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(f"/model/{file_name}")
                with fs.create(hdfs_path) as hdfs_file:
                    hdfs_file.write(buffer.getvalue())
        
        logger.info("Đã lưu mô hình tổng và scaler chung vào HDFS: /model")
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình chung: {e}")
        spark.stop()
        return

    spark.stop()
    logger.info("Hoàn tất huấn luyện mô hình chung!")

if __name__ == "__main__":
    split_train_total()