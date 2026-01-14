"""
Zoomex Exchange REST API Client Module

Implements the Zoomex V3 API for perpetual futures and spot trading.
API is similar to Bybit V5 API.

Supports:
- Linear perpetual futures (category: linear)
- Spot trading (category: spot) with conditional orders
"""
import json
import time
import hmac
import hashlib
import requests
from typing import Dict, Any, List, Optional, Tuple
from logger import setup_logger
from .base_client import BaseExchangeClient
from .proxy_utils import get_proxy_config

logger = setup_logger("api.zoomex")


class ZoomexClient(BaseExchangeClient):
    """Zoomex exchange client (REST).

    Implements Zoomex V3 API for USDT perpetual futures trading.
    Authentication uses HMAC-SHA256 with X-BAPI-* headers.
    """

    # API endpoints
    DEFAULT_BASE_URL = "https://openapi.zoomex.com"
    TESTNET_BASE_URL = "https://openapi-testnet.zoomex.com"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "") or config.get("secret_key", "")
        self.base_url = config.get("base_url", self.DEFAULT_BASE_URL)
        self.recv_window = config.get("recv_window", "5000")
        self.category = config.get("category", "linear")  # linear, inverse, spot
        self.position_mode = config.get("position_mode", "hedge")  # "hedge" or "one_way"

        # Proxy configuration
        self.proxies = get_proxy_config()
        if self.proxies:
            logger.info(f"Zoomex client configured with proxy: {self.proxies}")

    def get_exchange_name(self) -> str:
        return "Zoomex"

    async def connect(self) -> None:
        logger.info("Zoomex client connected")

    async def disconnect(self) -> None:
        logger.info("Zoomex client disconnected")

    # ------------------------------------------------------------------
    # Signature Generation
    # ------------------------------------------------------------------

    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """Generate HMAC-SHA256 signature for Zoomex API.
        
        Signature format: HMAC_SHA256(timestamp + api_key + recv_window + payload)
        """
        param_str = timestamp + self.api_key + self.recv_window + payload
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_auth_headers(self, payload: str = "") -> Dict[str, str]:
        """Generate authentication headers for Zoomex API."""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, payload)
        
        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': self.recv_window,
            'Content-Type': 'application/json'
        }

    # ------------------------------------------------------------------
    # HTTP Request Methods
    # ------------------------------------------------------------------

    def make_request(self, method: str, endpoint: str, api_key=None, secret_key=None,
                     instruction=None, params=None, data=None, retry_count: int = 3) -> Dict:
        """Execute HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            api_key: Not used (kept for interface compatibility)
            secret_key: Not used (kept for interface compatibility)
            instruction: Not used (kept for interface compatibility)
            params: Query parameters (for GET requests)
            data: Request body data (for POST requests)
            retry_count: Number of retries
            
        Returns:
            API response as dict
        """
        url = f"{self.base_url}{endpoint}"
        
        # Build payload string for signature
        if method.upper() == "GET" and params:
            payload = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        elif method.upper() == "POST" and data:
            payload = json.dumps(data, separators=(',', ':'))
        else:
            payload = ""

        # Get authentication headers
        headers = self._get_auth_headers(payload)

        # Add query params to URL for GET requests
        if method.upper() == "GET" and params:
            url += "?" + payload

        # Retry logic
        for attempt in range(retry_count):
            try:
                if method.upper() == "GET":
                    response = requests.get(
                        url, headers=headers,
                        proxies=self.proxies or None,
                        timeout=10
                    )
                elif method.upper() == "POST":
                    response = requests.post(
                        url, headers=headers,
                        data=payload if data else None,
                        proxies=self.proxies or None,
                        timeout=10
                    )
                elif method.upper() == "DELETE":
                    response = requests.delete(
                        url, headers=headers,
                        data=payload if data else None,
                        proxies=self.proxies or None,
                        timeout=10
                    )
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                # Parse response
                if response.status_code in [200, 201]:
                    result = response.json() if response.text.strip() else {}
                    
                    # Check for API-level errors
                    if result.get("retCode", 0) != 0:
                        error_msg = result.get("retMsg", "Unknown error")
                        return {"error": f"API error: {error_msg}", "raw": result}
                    
                    return result
                    
                elif response.status_code == 429:  # Rate limit
                    wait_time = 1 * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Status: {response.status_code}, Message: {response.text}"
                    if attempt < retry_count - 1:
                        logger.warning(f"Request failed ({attempt+1}/{retry_count}): {error_msg}")
                        time.sleep(1)
                        continue
                    return {"error": error_msg}

            except requests.exceptions.Timeout:
                if attempt < retry_count - 1:
                    logger.warning(f"Timeout ({attempt+1}/{retry_count}), retrying...")
                    continue
                return {"error": "Request timeout"}
            except requests.exceptions.ConnectionError:
                if attempt < retry_count - 1:
                    logger.warning(f"Connection error ({attempt+1}/{retry_count}), retrying...")
                    time.sleep(2)
                    continue
                return {"error": "Connection error"}
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Exception ({attempt+1}/{retry_count}): {str(e)}, retrying...")
                    continue
                return {"error": f"Request failed: {str(e)}"}

        return {"error": "Max retries reached"}

    # ------------------------------------------------------------------
    # Market Data Methods (Public)
    # ------------------------------------------------------------------

    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information for a symbol.
        
        Endpoint: GET /cloud/trade/v3/market/tickers
        """
        endpoint = "/cloud/trade/v3/market/tickers"
        params = {
            "category": self.category,
            "symbol": symbol
        }
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        # Parse ticker data
        result_data = response.get("result", {})
        ticker_list = result_data.get("list", [])
        
        if not ticker_list:
            return {"error": "No ticker data returned"}

        ticker = ticker_list[0]
        return {
            "symbol": ticker.get("symbol"),
            "lastPrice": ticker.get("lastPrice"),
            "bidPrice": ticker.get("bid1Price"),
            "askPrice": ticker.get("ask1Price"),
            "volume": ticker.get("volume24h"),
            "change24h": ticker.get("price24hPcnt"),
            "highPrice24h": ticker.get("highPrice24h"),
            "lowPrice24h": ticker.get("lowPrice24h"),
            "markPrice": ticker.get("markPrice"),
            "indexPrice": ticker.get("indexPrice"),
            "raw": ticker
        }

    def get_order_book(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook depth for a symbol.
        
        Endpoint: GET /cloud/trade/v3/market/orderbook
        """
        endpoint = "/cloud/trade/v3/market/orderbook"
        params = {
            "category": self.category,
            "symbol": symbol,
            "limit": str(limit)
        }
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        
        # Parse bids and asks: [[price, qty], ...]
        bids_raw = result.get("b", [])
        asks_raw = result.get("a", [])
        
        bids = [[float(b[0]), float(b[1])] for b in bids_raw]
        asks = [[float(a[0]), float(a[1])] for a in asks_raw]
        
        return {
            "symbol": result.get("s"),
            "bids": bids,
            "asks": asks,
            "timestamp": result.get("ts"),
            "updateId": result.get("u")
        }

    def get_klines(self, symbol: str, interval: str = "15", limit: int = 50) -> List[Dict]:
        """Get kline/candlestick data for ATR calculation.
        
        Endpoint: GET /cloud/trade/v3/market/kline
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            interval: Kline interval - "1", "3", "5", "15", "30", "60", "120", "240", "D", "W"
            limit: Number of candles (max 200, default 50)
            
        Returns:
            List of dicts with keys: open_time, open, high, low, close, volume, turnover
            Sorted in ascending order by time (oldest first).
        """
        endpoint = "/cloud/trade/v3/market/kline"
        
        # Map common interval formats to Zoomex format
        interval_mapping = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
            "1w": "W",
        }
        api_interval = interval_mapping.get(interval, interval)
        
        params = {
            "category": self.category,
            "symbol": symbol,
            "interval": api_interval,
            "limit": str(min(limit, 200))
        }
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response
        
        result = response.get("result", {})
        kline_list = result.get("list", [])
        
        # Parse klines: [[startTime, open, high, low, close, volume, turnover], ...]
        # API returns in reverse order (newest first), so we reverse to get oldest first
        klines = []
        for kline in reversed(kline_list):
            if len(kline) >= 6:
                klines.append({
                    "open_time": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "turnover": float(kline[6]) if len(kline) > 6 else 0.0,
                })
        
        return klines

    def get_markets(self) -> Dict:
        """Get all available trading instruments.
        
        Endpoint: GET /cloud/trade/v3/market/instruments-info
        """
        endpoint = "/cloud/trade/v3/market/instruments-info"
        params = {"category": self.category}
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        return {
            "category": result.get("category"),
            "list": result.get("list", [])
        }

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information including precision and limits.
        
        Args:
            symbol: Trading pair (e.g., "ETHUSDT")
            
        Returns:
            Dict with symbol info or None if not found.
        """
        params = {
            "category": "linear",
            "symbol": symbol
        }
        
        response = self.make_request(
            method="GET",
            endpoint="/cloud/trade/v3/market/instruments-info",
            params=params
        )
        
        if "error" in response:
            logger.error(f"Failed to get symbol info: {response['error']}")
            return None
            
        result = response.get("result", {})
        instruments = result.get("list", [])
        
        if not instruments:
            return None
            
        info = instruments[0]
        
        # Parse precision from filters
        lot_size = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})
        
        # Calculate precision from step sizes
        qty_step = lot_size.get("qtyStep", "0.001")
        tick_size = price_filter.get("tickSize", "0.01")
        
        # Count decimal places
        def count_decimals(value_str):
            if '.' in str(value_str):
                return len(str(value_str).split('.')[1].rstrip('0'))
            return 0
        
        base_precision = count_decimals(qty_step)
        quote_precision = count_decimals(tick_size)
        
        # Extract base/quote from symbol (e.g., ETHUSDT -> ETH, USDT)
        if symbol.endswith("USDT"):
            base_asset = symbol[:-4]
            quote_asset = "USDT"
        elif symbol.endswith("USDC"):
            base_asset = symbol[:-4]
            quote_asset = "USDC"
        else:
            base_asset = symbol
            quote_asset = "USD"
        
        return {
            "symbol": symbol,
            "base_asset": base_asset,
            "quote_asset": quote_asset,
            "base_precision": base_precision,
            "quote_precision": quote_precision,
            "tick_size": float(tick_size),
            "min_order_size": float(lot_size.get("minOrderQty", "0.001")),
            "max_order_size": float(lot_size.get("maxOrderQty", "1000")),
            "qty_step": float(qty_step),
            "raw": info
        }

    # ------------------------------------------------------------------
    # Account Methods (Private)
    # ------------------------------------------------------------------

    def get_balance(self) -> Dict:
        """Get wallet balance.
        
        Endpoint: GET /cloud/trade/v3/account/wallet-balance
        Returns dict keyed by asset name for compatibility with strategies.
        """
        endpoint = "/cloud/trade/v3/account/wallet-balance"
        params = {"accountType": "CONTRACT"}  # CONTRACT for perpetuals
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        account_list = result.get("list", [])
        
        if not account_list:
            return {}

        account = account_list[0]
        coins = account.get("coin", [])
        
        # Return dict keyed by asset name for compatibility with strategies
        balances = {}
        for coin in coins:
            asset = coin.get("coin")
            if asset:
                balances[asset] = {
                    "asset": asset,
                    "available": float(coin.get("availableToWithdraw", 0) or 0),
                    "locked": float(coin.get("locked", 0) or 0),
                    "total": float(coin.get("walletBalance", 0) or 0),
                    "equity": float(coin.get("equity", 0) or 0),
                    "unrealizedPnl": float(coin.get("unrealisedPnl", 0) or 0)
                }
        
        return balances

    def get_positions(self, symbol: Optional[str] = None) -> Any:
        """Get position information.
        
        Endpoint: GET /cloud/trade/v3/position/list
        Returns list of positions for compatibility with strategies.
        """
        endpoint = "/cloud/trade/v3/position/list"
        params = {"category": self.category}
        
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"  # Get all USDT positions
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        position_list = result.get("list", [])
        
        # Return list directly for compatibility with strategies
        positions = []
        for pos in position_list:
            size = float(pos.get("size", 0) or 0)
            if size > 0:  # Only include non-zero positions
                side = pos.get("side", "")
                # Normalize side to LONG/SHORT format
                if side.upper() == "BUY":
                    mapped_side = "LONG"
                    net_qty = size
                elif side.upper() == "SELL":
                    mapped_side = "SHORT"
                    net_qty = -size
                else:
                    mapped_side = side.upper() if side else "FLAT"
                    net_qty = size if mapped_side == "LONG" else -size
                
                positions.append({
                    "symbol": pos.get("symbol"),
                    "side": mapped_side,
                    "positionSide": mapped_side,
                    "size": str(size),
                    "netQuantity": str(net_qty),
                    "entryPrice": pos.get("avgPrice"),
                    "markPrice": pos.get("markPrice"),
                    "unrealizedPnl": pos.get("unrealisedPnl"),
                    "pnlUnrealized": pos.get("unrealisedPnl"),
                    "leverage": pos.get("leverage"),
                    "liquidationPrice": pos.get("liqPrice"),
                    "positionValue": pos.get("positionValue"),
                    "raw": pos
                })
        
        return positions

    def _get_position_idx(self, side: str, order_details: Dict) -> int:
        """Get the correct positionIdx based on position mode and order side.
        
        Args:
            side: Normalized order side ("Buy" or "Sell")
            order_details: Order details dict (may contain explicit positionIdx)
            
        Returns:
            positionIdx: 0 for one-way, 1 for hedge-long, 2 for hedge-short
            
        Hedge Mode Logic:
            - Open Long (Buy): positionIdx=1
            - Open Short (Sell): positionIdx=2
            - Close Long (Sell + reduceOnly): positionIdx=1 (same as Long position)
            - Close Short (Buy + reduceOnly): positionIdx=2 (same as Short position)
        """
        # Allow explicit override from order_details
        if "positionIdx" in order_details:
            return order_details["positionIdx"]
        
        # One-way mode: always use 0
        if self.position_mode == "one_way":
            return 0
        
        # Hedge mode: check if this is a closing order (reduceOnly)
        reduce_only = order_details.get("reduceOnly", False)
        
        if reduce_only:
            # Closing position: positionIdx matches the POSITION being closed
            # Sell to close Long -> positionIdx=1
            # Buy to close Short -> positionIdx=2
            return 1 if side == "Sell" else 2
        else:
            # Opening position: positionIdx matches the POSITION being opened
            # Buy to open Long -> positionIdx=1
            # Sell to open Short -> positionIdx=2
            return 1 if side == "Buy" else 2

    # ------------------------------------------------------------------
    # Order Methods (Private)
    # ------------------------------------------------------------------

    def execute_order(self, order_details: Dict) -> Dict:
        """Place a new order.
        
        Endpoint: POST /cloud/trade/v3/order/create
        
        Args:
            order_details: Dict with keys:
                - symbol: Trading pair (e.g., "BTCUSDT")
                - side: "Buy" or "Sell" (also accepts "Bid"/"Ask" which will be converted)
                - orderType: "Market" or "Limit"
                - qty: Order quantity (also accepts "quantity")
                - price: Order price (required for limit orders)
                - timeInForce: "GTC", "IOC", "FOK", "PostOnly"
                - positionIdx: 0 (one-way), 1 (hedge-buy), 2 (hedge-sell)
                - orderLinkId: Custom order ID (optional)
        """
        endpoint = "/cloud/trade/v3/order/create"
        
        # Normalize side: convert "Bid"/"Ask" to "Buy"/"Sell" for Zoomex API
        raw_side = order_details.get("side", "")
        side_mapping = {
            "Bid": "Buy",
            "bid": "Buy",
            "BID": "Buy",
            "Ask": "Sell",
            "ask": "Sell",
            "ASK": "Sell",
            "buy": "Buy",
            "BUY": "Buy",
            "sell": "Sell",
            "SELL": "Sell",
        }
        normalized_side = side_mapping.get(raw_side, raw_side)
        
        # Normalize quantity: accept both "qty" and "quantity"
        qty = order_details.get("qty") or order_details.get("quantity")
        
        # Build order payload
        data = {
            "category": self.category,
            "symbol": order_details.get("symbol"),
            "side": normalized_side,
            "orderType": order_details.get("orderType", "Limit"),
            "qty": str(qty),
            "positionIdx": self._get_position_idx(normalized_side, order_details)
        }
        
        # Optional fields
        if order_details.get("price"):
            data["price"] = str(order_details["price"])
        
        if order_details.get("timeInForce"):
            data["timeInForce"] = order_details["timeInForce"]
        else:
            data["timeInForce"] = "PostOnly"
        
        if order_details.get("orderLinkId"):
            data["orderLinkId"] = order_details["orderLinkId"]
        
        if order_details.get("reduceOnly"):
            data["reduceOnly"] = order_details["reduceOnly"]
        
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        order_id = result.get("orderId")
        return {
            "id": order_id,  # For strategy compatibility
            "orderId": order_id,
            "orderLinkId": result.get("orderLinkId"),
            "success": True,
            "raw": result
        }

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel a specific order.
        
        Endpoint: POST /cloud/trade/v3/order/cancel
        """
        endpoint = "/cloud/trade/v3/order/cancel"
        data = {
            "category": self.category,
            "symbol": symbol,
            "orderId": order_id
        }
        
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        return {
            "orderId": result.get("orderId"),
            "orderLinkId": result.get("orderLinkId"),
            "success": True,
            "raw": result
        }

    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all open orders.
        
        Endpoint: POST /cloud/trade/v3/order/cancel-all
        """
        endpoint = "/cloud/trade/v3/order/cancel-all"
        data = {"category": self.category}
        
        if symbol:
            data["symbol"] = symbol
        else:
            data["settleCoin"] = "USDT"  # Cancel all USDT perpetual orders
        
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        return {
            "success": True,
            "list": result.get("list", []),
            "raw": result
        }

    def get_open_orders(self, symbol: Optional[str] = None) -> Any:
        """Get open orders.
        
        Endpoint: GET /cloud/trade/v3/order/realtime
        Returns list of orders for compatibility with strategies.
        """
        endpoint = "/cloud/trade/v3/order/realtime"
        params = {
            "category": self.category,
            "openOnly": "0"  # Open orders only
        }
        
        if symbol:
            params["symbol"] = symbol
        else:
            params["settleCoin"] = "USDT"
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        order_list = result.get("list", [])
        
        # Return list directly for compatibility with strategies
        orders = []
        for order in order_list:
            order_id = order.get("orderId")
            orders.append({
                "id": order_id,  # For strategy compatibility
                "orderId": order_id,
                "orderLinkId": order.get("orderLinkId"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "orderType": order.get("orderType"),
                "price": order.get("price"),
                "qty": order.get("qty"),
                "filledQty": order.get("cumExecQty"),
                "status": order.get("orderStatus"),
                "timeInForce": order.get("timeInForce"),
                "createdTime": order.get("createdTime"),
                "raw": order
            })
        
        return orders

    def get_fill_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get trade execution history.
        
        Endpoint: GET /cloud/trade/v3/execution/list
        
        Args:
            symbol: Trading pair (optional)
            limit: Max number of records (default 50)
            
        Returns:
            List of fills for compatibility with strategies.
        """
        params = {
            "category": "linear",
            "limit": str(limit)
        }
        if symbol:
            params["symbol"] = symbol
            
        response = self.make_request(
            method="GET",
            endpoint="/cloud/trade/v3/execution/list",
            params=params
        )
        
        if "error" in response:
            return response
            
        fills = []
        result = response.get("result", {})
        fill_list = result.get("list", [])
        
        for fill in fill_list:
            # Normalize to strategy-expected format
            side = fill.get("side", "").upper()
            is_maker = fill.get("isMaker", False)
            
            fills.append({
                "id": fill.get("execId"),
                "fill_id": fill.get("execId"),
                "order_id": fill.get("orderId"),
                "symbol": fill.get("symbol"),
                "side": "BUY" if side == "BUY" else "SELL",
                "price": float(fill.get("execPrice", 0)),
                "quantity": float(fill.get("execQty", 0)),
                "size": float(fill.get("execQty", 0)),
                "is_maker": is_maker,
                "fee": float(fill.get("execFee", 0)),
                "fee_asset": fill.get("feeCurrency", "USDT"),
                "realized_pnl": float(fill.get("closedPnl", 0)),
                "timestamp": fill.get("execTime"),
                "raw": fill
            })
        
        return fills

    # ------------------------------------------------------------------
    # Leverage Management
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: int = 20) -> Dict:
        """Set leverage for a symbol.
        
        Endpoint: POST /cloud/trade/v3/position/set-leverage
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            leverage: Leverage value (1-100), default 20
            
        Returns:
            API response dict with success or error
        """
        # Clamp leverage to valid range
        leverage = max(1, min(100, leverage))
        leverage_str = str(leverage)
        
        payload = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": leverage_str,
            "sellLeverage": leverage_str,
        }
        
        logger.info(f"Setting leverage for {symbol} to {leverage}x")
        
        response = self.make_request(
            method="POST",
            endpoint="/cloud/trade/v3/position/set-leverage",
            data=payload
        )
        
        if "error" in response:
            logger.error(f"Failed to set leverage: {response}")
            return response
        
        logger.info(f"Successfully set leverage for {symbol} to {leverage}x")
        return {
            "success": True,
            "symbol": symbol,
            "leverage": leverage,
            "raw": response
        }

    def get_leverage(self, symbol: str) -> Dict:
        """Get current leverage for a symbol from position info.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            
        Returns:
            Dict with current leverage or error
        """
        positions = self.get_positions(symbol)
        
        if isinstance(positions, dict) and "error" in positions:
            return positions
        
        if positions and len(positions) > 0:
            pos = positions[0]
            return {
                "symbol": symbol,
                "leverage": pos.get("leverage", 1),
                "success": True
            }
        
        # No position, return default
        return {
            "symbol": symbol,
            "leverage": 20,  # Default Zoomex leverage
            "success": True,
            "note": "No active position, returning default leverage"
        }

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def get_market_limits(self, symbol: str) -> Optional[Dict]:
        """Get trading limits for a symbol (min order size, tick size, etc.)."""
        markets = self.get_markets()
        
        if "error" in markets:
            logger.error(f"Failed to get markets: {markets}")
            return None

        for instrument in markets.get("list", []):
            if instrument.get("symbol") == symbol:
                lot_filter = instrument.get("lotSizeFilter", {})
                price_filter = instrument.get("priceFilter", {})
                
                # Extract base and quote from symbol (e.g., BTCUSDT -> BTC, USDT)
                # Zoomex uses settleCoin as quote asset indicator
                quote_coin = instrument.get("settleCoin", "USDT")
                base_coin = instrument.get("baseCoin", symbol.replace(quote_coin, ""))
                
                # Calculate precision from step sizes
                qty_step = lot_filter.get("qtyStep", "0.001")
                tick_size = price_filter.get("tickSize", "0.01")
                
                # Count decimal places for precision
                base_precision = len(qty_step.split('.')[-1]) if '.' in qty_step else 0
                quote_precision = len(tick_size.split('.')[-1]) if '.' in tick_size else 0
                
                return {
                    "symbol": symbol,
                    "base_asset": base_coin,
                    "quote_asset": quote_coin,
                    "base_precision": base_precision,
                    "quote_precision": quote_precision,
                    "min_order_size": float(lot_filter.get("minOrderQty", "0.001")),
                    "max_order_size": float(lot_filter.get("maxOrderQty", "1000000")),
                    "tick_size": float(tick_size),
                    "qty_step": float(qty_step),
                    "raw": instrument
                }
        
        logger.error(f"Symbol {symbol} not found in markets")
        return None

    def get_deposit_address(self, blockchain: str) -> Dict:
        """Not implemented for Zoomex - returns error."""
        return {"error": "Deposit address query not supported for Zoomex"}

    def get_collateral(self, subaccount_id=None) -> Dict:
        """Get collateral info - alias for get_balance."""
        return self.get_balance()

    # ------------------------------------------------------------------
    # Spot Trading Methods
    # ------------------------------------------------------------------

    def get_spot_balance(self) -> Dict:
        """Get spot account balance.
        
        Endpoint: GET /cloud/trade/v3/account/wallet-balance
        Uses accountType: UNIFIED for spot trading.
        Returns dict keyed by asset name.
        """
        endpoint = "/cloud/trade/v3/account/wallet-balance"
        params = {"accountType": "UNIFIED"}  # UNIFIED for spot
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        account_list = result.get("list", [])
        
        if not account_list:
            return {}

        account = account_list[0]
        coins = account.get("coin", [])
        
        # Return dict keyed by asset name for compatibility with strategies
        balances = {}
        for coin in coins:
            asset = coin.get("coin")
            if asset:
                balances[asset] = {
                    "asset": asset,
                    "available": float(coin.get("availableToWithdraw", 0) or 0),
                    "locked": float(coin.get("locked", 0) or 0),
                    "total": float(coin.get("walletBalance", 0) or 0),
                    "free": float(coin.get("free", 0) or coin.get("availableToWithdraw", 0) or 0),
                }
        
        return balances

    def get_spot_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get spot symbol information including precision and limits.
        
        Args:
            symbol: Trading pair (e.g., "MNTUSDT")
            
        Returns:
            Dict with symbol info or None if not found.
        """
        params = {
            "category": "spot",
            "symbol": symbol
        }
        
        response = self.make_request(
            method="GET",
            endpoint="/cloud/trade/v3/market/instruments-info",
            params=params
        )
        
        if "error" in response:
            logger.error(f"Failed to get spot symbol info: {response['error']}")
            return None
            
        result = response.get("result", {})
        instruments = result.get("list", [])
        
        if not instruments:
            return None
            
        info = instruments[0]
        
        # Parse precision from filters
        lot_size = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})
        
        # Calculate precision from step sizes
        qty_step = lot_size.get("basePrecision", lot_size.get("qtyStep", "0.0001"))
        tick_size = price_filter.get("tickSize", "0.0001")
        
        # Count decimal places
        def count_decimals(value_str):
            if '.' in str(value_str):
                return len(str(value_str).split('.')[1].rstrip('0'))
            return 0
        
        base_precision = count_decimals(qty_step)
        quote_precision = count_decimals(tick_size)
        
        return {
            "symbol": symbol,
            "base_asset": info.get("baseCoin", symbol[:-4] if symbol.endswith("USDT") else symbol),
            "quote_asset": info.get("quoteCoin", "USDT"),
            "base_precision": base_precision,
            "quote_precision": quote_precision,
            "tick_size": float(tick_size),
            "min_order_size": float(lot_size.get("minOrderQty", "0.0001")),
            "max_order_size": float(lot_size.get("maxOrderQty", "1000000")),
            "qty_step": float(qty_step),
            "raw": info
        }

    def execute_spot_order(self, order_details: Dict) -> Dict:
        """Place a spot order (regular or conditional).
        
        Endpoint: POST /cloud/trade/v3/order/create
        
        Args:
            order_details: Dict with keys:
                - symbol: Trading pair (e.g., "MNTUSDT")
                - side: "Buy" or "Sell"
                - orderType: "Market" or "Limit"
                - qty: Order quantity
                - price: Order price (required for limit orders)
                - timeInForce: "GTC", "IOC", "FOK", "PostOnly"
                - orderLinkId: Custom order ID (optional)
                
                For conditional orders (orderFilter: "StopOrder"):
                - triggerPrice: Price that activates the order
                - triggerDirection: 1 (rise to) or 2 (fall to)
                - orderFilter: "StopOrder" for conditional, "Order" for regular
        """
        endpoint = "/cloud/trade/v3/order/create"
        
        # Normalize side: convert "Bid"/"Ask" to "Buy"/"Sell"
        raw_side = order_details.get("side", "")
        side_mapping = {
            "Bid": "Buy", "bid": "Buy", "BID": "Buy",
            "Ask": "Sell", "ask": "Sell", "ASK": "Sell",
            "buy": "Buy", "BUY": "Buy",
            "sell": "Sell", "SELL": "Sell",
        }
        normalized_side = side_mapping.get(raw_side, raw_side)
        
        # Normalize quantity: accept both "qty" and "quantity"
        qty = order_details.get("qty") or order_details.get("quantity")
        
        # Build order payload
        data = {
            "category": "spot",
            "symbol": order_details.get("symbol"),
            "side": normalized_side,
            "orderType": order_details.get("orderType", "Limit"),
            "qty": str(qty),
        }
        
        # Price (required for limit orders)
        if order_details.get("price"):
            data["price"] = str(order_details["price"])
        
        # Time in force
        if order_details.get("timeInForce"):
            data["timeInForce"] = order_details["timeInForce"]
        else:
            data["timeInForce"] = "GTC"
        
        # Order filter: "Order" for regular, "StopOrder" for conditional
        order_filter = order_details.get("orderFilter", "Order")
        data["orderFilter"] = order_filter
        
        # Conditional order parameters
        if order_filter == "StopOrder":
            if order_details.get("triggerPrice"):
                data["triggerPrice"] = str(order_details["triggerPrice"])
            if order_details.get("triggerDirection"):
                data["triggerDirection"] = order_details["triggerDirection"]
        
        # Custom order ID
        if order_details.get("orderLinkId"):
            data["orderLinkId"] = order_details["orderLinkId"]
        
        logger.debug(f"Placing spot order: {data}")
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        order_id = result.get("orderId")
        return {
            "id": order_id,
            "orderId": order_id,
            "orderLinkId": result.get("orderLinkId"),
            "success": True,
            "raw": result
        }

    def get_spot_open_orders(self, symbol: Optional[str] = None, order_filter: str = "Order") -> Any:
        """Get open spot orders.
        
        Endpoint: GET /cloud/trade/v3/order/realtime
        
        Args:
            symbol: Trading pair (optional)
            order_filter: "Order" for regular, "StopOrder" for conditional
            
        Returns:
            List of orders for compatibility with strategies.
        """
        endpoint = "/cloud/trade/v3/order/realtime"
        params = {
            "category": "spot",
            "orderFilter": order_filter,
            "openOnly": "0"  # Open orders only
        }
        
        if symbol:
            params["symbol"] = symbol
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        order_list = result.get("list", [])
        
        # Return list directly for compatibility with strategies
        orders = []
        for order in order_list:
            order_id = order.get("orderId")
            orders.append({
                "id": order_id,
                "orderId": order_id,
                "orderLinkId": order.get("orderLinkId"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "orderType": order.get("orderType"),
                "price": order.get("price"),
                "qty": order.get("qty"),
                "filledQty": order.get("cumExecQty"),
                "status": order.get("orderStatus"),
                "timeInForce": order.get("timeInForce"),
                "triggerPrice": order.get("triggerPrice"),
                "orderFilter": order.get("orderFilter"),
                "createdTime": order.get("createdTime"),
                "raw": order
            })
        
        return orders

    def cancel_spot_order(self, order_id: str, symbol: str, order_filter: str = "Order") -> Dict:
        """Cancel a specific spot order.
        
        Endpoint: POST /cloud/trade/v3/order/cancel
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair
            order_filter: "Order" for regular, "StopOrder" for conditional
        """
        endpoint = "/cloud/trade/v3/order/cancel"
        data = {
            "category": "spot",
            "symbol": symbol,
            "orderId": order_id,
            "orderFilter": order_filter
        }
        
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        return {
            "orderId": result.get("orderId"),
            "orderLinkId": result.get("orderLinkId"),
            "success": True,
            "raw": result
        }

    def cancel_all_spot_orders(self, symbol: str = None, order_filter: str = "Order") -> Dict:
        """Cancel all open spot orders.
        
        Endpoint: POST /cloud/trade/v3/order/cancel-all
        
        Args:
            symbol: Trading pair (optional)
            order_filter: "Order" for regular, "StopOrder" for conditional
        """
        endpoint = "/cloud/trade/v3/order/cancel-all"
        data = {
            "category": "spot",
            "orderFilter": order_filter
        }
        
        if symbol:
            data["symbol"] = symbol
        
        response = self.make_request("POST", endpoint, data=data)
        
        if "error" in response:
            return response

        result = response.get("result", {})
        return {
            "success": True,
            "list": result.get("list", []),
            "raw": result
        }

    def get_spot_ticker(self, symbol: str) -> Dict:
        """Get spot ticker information for a symbol.
        
        Endpoint: GET /cloud/trade/v3/market/tickers
        """
        endpoint = "/cloud/trade/v3/market/tickers"
        params = {
            "category": "spot",
            "symbol": symbol
        }
        
        response = self.make_request("GET", endpoint, params=params)
        
        if "error" in response:
            return response

        # Parse ticker data
        result_data = response.get("result", {})
        ticker_list = result_data.get("list", [])
        
        if not ticker_list:
            return {"error": "No ticker data returned"}

        ticker = ticker_list[0]
        return {
            "symbol": ticker.get("symbol"),
            "lastPrice": ticker.get("lastPrice"),
            "bidPrice": ticker.get("bid1Price"),
            "askPrice": ticker.get("ask1Price"),
            "volume": ticker.get("volume24h"),
            "change24h": ticker.get("price24hPcnt"),
            "highPrice24h": ticker.get("highPrice24h"),
            "lowPrice24h": ticker.get("lowPrice24h"),
            "raw": ticker
        }

    def get_spot_fill_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get spot trade execution history.
        
        Endpoint: GET /cloud/trade/v3/execution/list
        
        Args:
            symbol: Trading pair (optional)
            limit: Max number of records (default 50)
            
        Returns:
            List of fills for compatibility with strategies.
        """
        params = {
            "category": "spot",
            "limit": str(limit)
        }
        if symbol:
            params["symbol"] = symbol
            
        response = self.make_request(
            method="GET",
            endpoint="/cloud/trade/v3/execution/list",
            params=params
        )
        
        if "error" in response:
            return response
            
        fills = []
        result = response.get("result", {})
        fill_list = result.get("list", [])
        
        for fill in fill_list:
            side = fill.get("side", "").upper()
            is_maker = fill.get("isMaker", False)
            
            fills.append({
                "id": fill.get("execId"),
                "fill_id": fill.get("execId"),
                "order_id": fill.get("orderId"),
                "symbol": fill.get("symbol"),
                "side": "BUY" if side == "BUY" else "SELL",
                "price": float(fill.get("execPrice", 0)),
                "quantity": float(fill.get("execQty", 0)),
                "size": float(fill.get("execQty", 0)),
                "is_maker": is_maker,
                "fee": float(fill.get("execFee", 0)),
                "fee_asset": fill.get("feeCurrency", ""),
                "timestamp": fill.get("execTime"),
                "raw": fill
            })
        
        return fills

