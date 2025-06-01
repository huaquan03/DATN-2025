from flask import Flask, request, jsonify
import requests

import logging


# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/update-stock', methods=['POST'])
def update_stock():
    # Nhận dữ liệu từ Grafana
    data = request.get_json()
    ticker = data.get('ticker')
    logger.info(f"da nhan request tu Grafana: {ticker}")

    if not ticker:
        return jsonify({"status": "error", "message": "Ticker is required"}), 400

    # Chuyển tiếp yêu cầu POST đến container crawler
    crawler_url = f"http://crawler-realtime:5000/update-stock"
    try:
        logger.info(f"Sending request to crawler for ticker: {ticker}")
        response = requests.post(crawler_url, json={"ticker": ticker}, timeout=10)
        if response.status_code == 200:
            logger.info("Request to crawler successful")
            return jsonify({"status": "success", "message": f"Data for {ticker} updated"})  
        else:
            logger.error(f"Request to crawler failed with status: {response.status_code}")
            return jsonify({"status": "error", "message": f"Failed to crawl data: {response.text}"}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Crawler request failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Crawler request failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)