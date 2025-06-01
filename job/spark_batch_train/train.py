import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pyspark.sql import SparkSession
import joblib
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lstm(X, y, timesteps=30, epochs=50, batch_size=32, learning_rate=0.0001, fine_tune=False):
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
        if fine_tune:
            model.fit(X_reshaped, y_reshaped, epochs=20, batch_size=batch_size,
                     callbacks=[early_stopping], verbose=1)
        else:
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

def train_models():
    # Thêm độ trễ để đảm bảo Namenode sẵn sàng
    logger.info("Đợi 10 giây để Namenode sẵn sàng...")
    time.sleep(10)
    
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("VN30ModelTraining") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "6") \
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
        return None, None, None
    
    # Kiểm tra dữ liệu
    logger.info("Số dòng mỗi mã cổ phiếu:")
    logger.info(df.groupby('ticker').size())
    
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
        return None, None, None
    
    # Bước 1: Huấn luyện mô hình chung
    logger.info("\nHuấn luyện mô hình chung cho tất cả target")
    try:
        X = df[features]
        y = df['target_close_t+1']
        if len(X) < 100:
            logger.error(f"Dữ liệu quá ngắn ({len(X)} dòng)")
            spark.stop()
            return None, None, None
        model_common, scaler_X_common, scaler_y_common = train_lstm(X, y, timesteps=30, epochs=50)
        
        # Lưu mô hình tổng và scaler chung cục bộ
        model_common.save("temp_common_model.h5")
        joblib.dump(scaler_X_common, "temp_scaler_X_common.pkl")
        joblib.dump(scaler_y_common, "temp_scaler_y_common.pkl")
        
        # Upload lên HDFS bằng Spark
        for file in ["temp_common_model.h5", "temp_scaler_X_common.pkl", "temp_scaler_y_common.pkl"]:
            spark.sparkContext.binaryFiles(f"file://{os.getcwd()}/{file}") \
                .saveAsTextFile(f"hdfs://namenode:9000/model/{file}")
        logger.info("Đã lưu mô hình tổng và scaler chung vào /model")
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình chung: {e}")
        spark.stop()
        return None, None, None
    
    # Bước 2: Fine-tune mô hình cho từng mã
    models = {}
    scalers = {}
    def fine_tune_ticker(ticker):
        ticker_df = df[df['ticker'] == ticker]
        ticker_models = {}
        ticker_scalers = {}
        ticker_dir = f"ticker={ticker}"
        os.makedirs(ticker_dir, exist_ok=True)
        hdfs_dir = f"/model/ticker={ticker}"
        
        for target in targets:
            X = ticker_df[features]
            y = ticker_df[target]
            if len(X) < 100:
                logger.warning(f"Bỏ qua {ticker} - {target}: dữ liệu ngắn ({len(X)} dòng)")
                continue
            try:
                model = clone_model(model_common)
                model.set_weights(model_common.get_weights())
                model.compile(optimizer=Adam(learning_rate=0.00005), loss='mse')
                model, scaler_X, scaler_y = train_lstm(X, y, timesteps=30, epochs=20, fine_tune=True)
                
                # Lưu cục bộ tạm thời
                model.save(f"{ticker_dir}/model_finetune_{target}.h5")
                joblib.dump(scaler_X, f"{ticker_dir}/scaler_X_{target}.pkl")
                joblib.dump(scaler_y, f"{ticker_dir}/scaler_y_{target}.pkl")
                
                # Upload lên HDFS bằng Spark
                for file in [f"model_finetune_{target}.h5", f"scaler_X_{target}.pkl", f"scaler_y_{target}.pkl"]:
                    spark.sparkContext.binaryFiles(f"file://{os.getcwd()}/{ticker_dir}/{file}") \
                        .saveAsTextFile(f"hdfs://namenode:9000/{hdfs_dir}/{file}")
                
                ticker_models[(ticker, target)] = model
                ticker_scalers[(ticker, target)] = (scaler_X, scaler_y)
                logger.info(f"Fine-tuning cho {ticker} - {target} hoàn tất, lưu vào {hdfs_dir}")
            except Exception as e:
                logger.error(f"Lỗi khi fine-tune cho {ticker} - {target}: {e}")
                continue
        return ticker_models, ticker_scalers
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fine_tune_ticker, df['ticker'].unique())
        for ticker_models, ticker_scalers in results:
            models.update(ticker_models)
            scalers.update(ticker_scalers)
    
    # Dọn dẹp cục bộ
    for ticker in df['ticker'].unique():
        ticker_dir = f"ticker={ticker}"
        for target in targets:
            for file in [f"model_finetune_{target}.h5", f"scaler_X_{target}.pkl", f"scaler_y_{target}.pkl"]:
                if os.path.exists(f"{ticker_dir}/{file}"):
                    os.remove(f"{ticker_dir}/{file}")
        if os.path.exists(ticker_dir):
            os.rmdir(ticker_dir)
    for file in ["temp_common_model.h5", "temp_scaler_X_common.pkl", "temp_scaler_y_common.pkl"]:
        if os.path.exists(file):
            os.remove(file)
    
    spark.stop()
    return models, scalers, features

if __name__ == "__main__":
    models, scalers, features = train_models()