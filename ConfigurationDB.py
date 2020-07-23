username = 'userStockPrice'
password = 'user'
nameDB = 'stockprice'
keyApi = 'pythonKeyApi'
sqlSELECT = "SELECT date, price_close FROM historical_price_companies WHERE symbol = %s AND date >= %s AND date <= %s ORDER BY date ASC"
sqlSELECT2 = "SELECT date, price_close FROM historical_price_companies WHERE symbol = %s AND date >= %s AND date <= %s ORDER BY date  DESC LIMIT %s"
