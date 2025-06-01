from vnstock import Vnstock
import pandas as pd
from datetime import datetime
from datetime import datetime, timedelta
end = datetime.now().parse('%Y-%m-%d')
start=end-timedelta(days=365)

print(start)
print(end)