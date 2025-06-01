import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Feature engineering cho hdfs 
def feature_engineering_for_hdfs(df):
    try:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.sort_values('time')
        df['time_idx'] = np.arange(len(df))
        df['ref_price'] = df['close'].shift(1)
        df['ceiling_price'] = df['ref_price'] * 1.07
        df['floor_price'] = df['ref_price'] * 0.93
        epsilon = 0.01
        df['ceiling_floor'] = 0
        df.loc[df['close'] >= df['ceiling_price'] - epsilon, 'ceiling_floor'] = 1
        df.loc[df['close'] <= df['floor_price'] + epsilon, 'ceiling_floor'] = -1
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['price_change_pct'] = df['price_change_pct'].replace([np.inf, -np.inf], np.nan)
        df['historical_volatility_20'] = df['price_change_pct'].rolling(window=20).std()
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_change_pct'] = df['volume'].pct_change() * 100
        df['volume_change_pct'] = df['volume_change_pct'].replace([np.inf, -np.inf], np.nan)
        df['sma5'] = SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma5_pct'] = np.where(df['close'] != 0, ((df['close'] - df['sma5']) / df['close']) * 100, np.nan)
        df['sma20_pct'] = np.where(df['close'] != 0, ((df['close'] - df['sma20']) / df['close']) * 100, np.nan)
        df['sma50_pct'] = np.where(df['close'] != 0, ((df['close'] - df['sma50']) / df['close']) * 100, np.nan)
        df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_high_pct'] = np.where(df['close'] != 0, ((df['bb_high'] - df['close']) / df['close']) * 100, np.nan)
        df['bb_low_pct'] = np.where(df['close'] != 0, ((df['bb_low'] - df['close']) / df['close']) * 100, np.nan)
        df['bb_mid_pct'] = np.where(df['close'] != 0, ((df['bb_mid'] - df['close']) / df['close']) * 100, np.nan)
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_pct'] = np.where(df['close'] != 0, (df['macd'] / df['close']) * 100, np.nan)
        df['macd_signal_pct'] = np.where(df['close'] != 0, (df['macd_signal'] / df['close']) * 100, np.nan)
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['days_since_start'] = (df['time'] - df['time'].min()).dt.days
        for lag in [1, 3, 5]:
            df[f'close_lag{lag}_pct'] = np.where(df['close'] != 0, ((df['close'] - df['close'].shift(lag)) / df['close']) * 100, np.nan)
            df[f'volume_lag{lag}'] = df['volume'].shift(lag)
        # Thêm cột target cho việc train model
        df['target_close_t+1'] = df['close'].shift(-1)
        df['target_close_t+3'] = df['close'].shift(-3)
        df['target_close_t+5'] = df['close'].shift(-5)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
 
        return df
    except Exception as e:
        logger.error(f"Lỗi trong feature_engineering_for_hdfs: {e}")
        return None


# Hàm đẩy dữ liệu vào HDFS

def push_to_hdfs(spark, df, hdfs_path="hdfs://namenode:9000/data/vn30_features.parquet"):
    try:
        if df.empty:
            logger.error("Dữ liệu đầu vào rỗng, không thể đẩy vào HDFS")
            return False
        
        # Chuyển DataFrame pandas thành Spark DataFrame
        spark_df = spark.createDataFrame(df)
        
        # Lưu vào HDFS
        spark_df.write \
            .partitionBy("ticker") \
            .mode("overwrite") \
            .parquet(hdfs_path)
        logger.info(f"Dữ liệu đã được ghi vào HDFS tại: {hdfs_path}")

        return True
    except Exception as e:
        logger.error(f"Lỗi khi đẩy dữ liệu vào HDFS: {e}")
        return False
    