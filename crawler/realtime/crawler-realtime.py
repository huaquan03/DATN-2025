from fastapi import FastAPI, HTTPException
from vnstock import Vnstock
import pandas as pd
from datetime import datetime,timedelta
from kafka import KafkaProducer

app = FastAPI()

# Hàm fetch_data (giữ nguyên logic của bạn)
def fetch_data(symbol):
    stock = Vnstock().stock(symbol=symbol, source='VCI')
    end = datetime.now()
    # end =end -timedelta(days=3)  
    end=end.strftime('%Y-%m-%d')
    df_pandas = stock.quote.history(start=end, end=end, interval='1m')
    df_pandas['ticker'] = symbol
    df_pandas = df_pandas.rename(columns={'date': 'time'})
    print(df_pandas)
    return df_pandas.to_json(orient='records', date_format='iso')

# Tạo Kafka Producer (khởi tạo một lần khi ứng dụng chạy)
producer = KafkaProducer(
    bootstrap_servers=['kafka1:9093', 'kafka2:9093', 'kafka3:9093'],
    value_serializer=lambda v: v.encode('utf-8')
)

@app.post("/update-stock")
async def update_stock(data: dict):
    ticker = data.get('ticker')
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    try:
        # Thu thập dữ liệu bằng hàm fetch_data
        data_json = fetch_data(ticker)
        print(f"Đang gửi dữ liệu cho {ticker}: {data_json}")

        # Gửi dữ liệu vào Kafka
        producer.send('GIACKREALTIME', value=data_json)
        producer.flush()  # Đảm bảo dữ liệu được gửi hết

        return {"status": "success", "message": f"Data for {ticker} crawled and sent to Kafka"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error crawling data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)