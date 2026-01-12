"""Test the get_klines method for ZoomexClient"""
import sys
sys.path.insert(0, '.')

from api.zoomex_client import ZoomexClient

# Create client (klines is public endpoint, no auth needed)
client = ZoomexClient({'api_key': 'test', 'api_secret': 'test'})

print("Testing get_klines for ETHUSDT 15m...")
klines = client.get_klines('ETHUSDT', '15m', 20)

if isinstance(klines, dict) and 'error' in klines:
    print(f"ERROR: {klines}")
elif isinstance(klines, list):
    print(f"SUCCESS: Got {len(klines)} klines")
    if klines:
        print(f"First kline: {klines[0]}")
        print(f"Last kline: {klines[-1]}")
        
        # Calculate sample ATR
        if len(klines) >= 14:
            trs = []
            for i in range(1, len(klines)):
                high = klines[i]['high']
                low = klines[i]['low']
                prev_close = klines[i-1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                trs.append(tr)
            
            atr_14 = sum(trs[-14:]) / 14
            natr = (atr_14 / klines[-1]['close']) * 100
            print(f"\nSample ATR(14): {atr_14:.4f}")
            print(f"Sample NATR(14): {natr:.4f}%")
else:
    print(f"Unexpected result type: {type(klines)}")
