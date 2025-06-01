import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import logging
import clickhouse_connect

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Feature enigineering cho ClickHouse
def feature_engineering_for_clickhouse(df):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        df['time'] = pd.to_datetime(df['time'])
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.sort_values('time')
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['sma5'] = SMAIndicator(df['close'], window=5).sma_indicator()
        df['sma20'] = SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator()
        df['rsi14'] = RSIIndicator(df['close'], window=14).rsi()
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
 
        return df
    except Exception as e:
        logger.error(f"Lỗi trong feature_engineering_for_clickhouse: {e}")
        return None


# Hàm đẩy dữ liệu vào ClickHouse

def push_to_clickhouse(df, table_name="gialichsu", host="clickhouse01", port=8123, username="default"):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df['time'] = pd.to_datetime(df['time'])
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        if df.empty:
            logger.error("Dữ liệu sau lọc rỗng")
            return False
        else :
            logger.info(f"Dữ liệu sau lọc có {len(df)} bản ghi")
        # 2. Kết nối ClickHouse
        client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            database='default',
            settings={
                'max_block_size': 1000,
                'connect_timeout': 10,
            }
        )

        # 3. Kiểm tra cụm
        clusters = client.query("SELECT cluster FROM system.clusters").result_rows
        if not any('my_cluster' in row for row in clusters):
            logger.error("Cụm 'my_cluster' không tồn tại")
            return False
        logger.info("Cụm 'my_cluster' đã được tìm thấy")

        # 4. Xóa bảng cũ
        client.command("DROP TABLE IF EXISTS default.gialichsu ON CLUSTER my_cluster SYNC")

        # 5. Tạo bảng ReplicatedMergeTree với schema không có cột target
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS default.gialichsu ON CLUSTER my_cluster (
                time DateTime,
                open Float64,
                high Float64,
                low Float64,
                close Float64,
                volume Int64,
                ticker String,
                volume_ma5 Float64,
                volume_ma20 Float64,
                sma5 Float64,
                sma20 Float64,
                sma50 Float64,
                rsi14 Float64,
                bb_high Float64,
                bb_low Float64,
                bb_mid Float64,
                macd Float64,
                macd_signal Float64,

            ) ENGINE = ReplicatedMergeTree('/clickhouse/tables/shard1/gialichsu', '{replica}')
            ORDER BY (ticker, time)
            PARTITION BY toYYYYMM(time)
            SETTINGS index_granularity = 8192
        '''
        client.command(create_table_sql)


        client.insert_df(table_name, df)
        logger.info(f"Đã đẩy thành công {len(df)} bản ghi vào ClickHouse bảng {table_name}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi đẩy dữ liệu vào ClickHouse: {e}")
        return False
    