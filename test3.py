
from vnstock import Vnstock
import pandas as pd
from datetime import datetime, timedelta

# Lấy danh sách mã trong nhóm VN30
df_symbols = Vnstock().stock(symbol='ACB', source='VCI').listing.symbols_by_group('VN30')

# Xác định ngày hiện tại dưới dạng chuỗi YYYY-MM-DD
today = datetime.now().date()
date_str = today.strftime('%Y-%m-%d')
# Thời điểm ngưỡng 15 phút trước
threshold = datetime.now() - timedelta(minutes=15)

all_data = []
for symbol in df_symbols:
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    # Lấy dữ liệu cả ngày với interval 1 phút
    df = stock.quote.history(start=date_str, end=date_str, interval='1m')
    # Đổi tên cột date thành time
    df = df.rename(columns={'date': 'time'})
    # Chuyển cột time sang datetime
    df['time'] = pd.to_datetime(df['time'])
    # Lọc chỉ giữ 15 phút gần nhất
    df = df[df['time'] >= threshold]
    if not df.empty:
        df['ticker'] = symbol
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(combined_df)
else:
    print("Không có dữ liệu trong 15 phút gần nhất.")
