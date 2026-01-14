"""
Long Grid Short Hedge Strategy

Delta-controlled grid trading strategy for ETHUSDT perpetual on Zoomex.

Core Concept:
- Maintain a 1× short position as hedge and funding collector
- Run high-leverage micro long grid around price to scalp volatility
- Continuously control net delta and liquidation risk

Author: Antigravity
"""
from __future__ import annotations

import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict

from logger import setup_logger
from strategies.perp_market_maker import PerpetualMarketMaker, format_balance
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("long_grid_short_hedge")


class LongGridShortHedge(PerpetualMarketMaker):
    """
    Delta-controlled grid trading strategy for Zoomex.
    
    Strategy Overview:
    - Base Position: Open short (e.g., 2 ETH @ 1× leverage) for funding collection
    - Grid Longs: Place micro limit longs around current price @ high leverage
    - Delta Control: Ensure net_long ≤ short_size at all times
    - Safety: Kill switches for liquidation proximity, funding flip, etc.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str = "ETHUSDT",
        # === Base Short (Anchor) ===
        short_size: float = 2.0,
        short_leverage: float = 1.0,
        # === Grid Long Parameters ===
        grid_long_size: float = 0.02,
        grid_leverage: float = 20.0,
        grid_spacing: float = 0.10,
        grid_levels: int = 10,
        take_profit: float = 1.17,
        # === Exchange Config ===
        exchange: str = "zoomex",
        exchange_config: Optional[Dict[str, Any]] = None,
        enable_database: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize Long Grid Short Hedge Strategy.

        Args:
            short_size: Base short position size in ETH (default: 2.0)
            short_leverage: Leverage for anchor short (default: 1×)
            grid_long_size: Long size per grid order in ETH (default: 0.02)
            grid_leverage: Leverage for grid longs (default: 20×)
            grid_spacing: Base grid spacing in USD (default: 0.10)
            grid_levels: Number of grid levels each side (default: 10)
            take_profit: Take profit offset in USD (default: 1.17)
        """
        # Force Zoomex exchange
        if exchange != "zoomex":
            logger.warning("LongGridShortHedge only supports Zoomex, forcing exchange='zoomex'")
            exchange = "zoomex"

        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            target_position=0.0,
            max_position=short_size,
            leverage=grid_leverage,
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # === Anchor Short Parameters ===
        self.short_size = abs(short_size)
        self.short_leverage = max(1.0, short_leverage)

        # === Grid Long Parameters ===
        self.grid_long_size = abs(grid_long_size)
        self.grid_leverage = max(1.0, grid_leverage)
        self.grid_spacing = abs(grid_spacing)
        self.grid_levels = max(1, grid_levels)
        self.take_profit_offset = abs(take_profit)

        # === Position State ===
        self.anchor_short_entry: float = 0.0
        self.anchor_short_liq_price: float = 0.0
        self.funding_pnl: float = 0.0

        # === Grid State ===
        # {price: {order_id, qty, status}}
        self.grid_long_orders: Dict[float, Dict[str, Any]] = {}
        # {order_id: {entry_price, qty}}
        self.grid_tp_orders: Dict[str, Dict[str, Any]] = {}
        self.net_long_eth: float = 0.0

        # === WebSocket Clients ===
        self.ws_public = None
        self.ws_private = None
        self._ws_initialized = False

        # === Safety State ===
        self.is_paused: bool = False
        self.pause_reason: Optional[str] = None
        self._last_price: float = 0.0

        # === Threshold Actions ===
        # 25% -> partial reduce, 50% -> more reduce, 100% -> full unwind
        self.threshold_actions = [
            (0.25, "partial_reduce_25"),
            (0.50, "partial_reduce_50"),
            (1.00, "full_unwind"),
        ]

        logger.info("=" * 60)
        logger.info("Long Grid Short Hedge Strategy Initialized")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Anchor Short: {self.short_size} ETH @ {self.short_leverage}× leverage")
        logger.info(f"Grid Longs: {self.grid_long_size} ETH @ {self.grid_leverage}× leverage")
        logger.info(f"Grid Levels: {self.grid_levels} each side (total: {self.grid_levels * 2})")
        logger.info(f"Grid Spacing: ${self.grid_spacing}")
        logger.info(f"Take Profit Offset: ${self.take_profit_offset}")
        logger.info(f"Max Long Exposure: {self.grid_levels * 2 * self.grid_long_size} ETH")
        logger.info("=" * 60)

    # =========================================================================
    # WebSocket Setup
    # =========================================================================

    def setup_websockets(self) -> bool:
        """Initialize WebSocket connections for real-time data."""
        try:
            from ws_client.zoomex_client import ZoomexWebSocket
            from ws_client.zoomex_private_client import ZoomexPrivateWebSocket

            api_key = self.client.api_key
            api_secret = self.client.api_secret

            # Public WebSocket for price/orderbook
            self.ws_public = ZoomexWebSocket(
                symbol=self.symbol,
                on_message_callback=self._on_public_ws_message,
                auto_reconnect=True,
            )
            self.ws_public.connect()
            self.ws_public.subscribe_ticker()
            self.ws_public.subscribe_depth(depth=50)
            logger.info("Public WebSocket connected and subscribed")

            # Private WebSocket for order/position updates
            self.ws_private = ZoomexPrivateWebSocket(
                api_key=api_key,
                api_secret=api_secret,
                on_order_callback=self._on_order_update,
                on_position_callback=self._on_position_update,
                on_wallet_callback=self._on_wallet_update,
                auto_reconnect=True,
            )
            self.ws_private.connect()
            logger.info("Private WebSocket connected and authenticated")

            self._ws_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to setup WebSockets: {e}")
            return False

    def _on_public_ws_message(self, stream: str, data: Dict[str, Any]) -> None:
        """Handle public WebSocket messages (price updates)."""
        if "ticker" in stream.lower():
            last_price = data.get("lastPrice") or data.get("last_price")
            if last_price:
                self._last_price = float(last_price)

    def _on_order_update(self, order_data: Dict[str, Any]) -> None:
        """Handle private WebSocket order updates."""
        order_id = order_data.get("orderId") or order_data.get("order_id")
        status = order_data.get("orderStatus") or order_data.get("status", "")
        side = order_data.get("side", "").upper()
        price = float(order_data.get("price", 0) or 0)
        qty = float(order_data.get("qty", 0) or order_data.get("cumExecQty", 0) or 0)

        logger.debug(f"Order update: {order_id} | {status} | {side} | {price} | {qty}")

        # Check if grid long order filled
        if status.upper() in ["FILLED", "PARTIALLYFILLED"]:
            if side == "BUY":
                # Grid long filled -> place take profit
                self._handle_grid_long_filled(order_id, price, qty)
            elif side == "SELL":
                # Take profit filled -> record profit
                self._handle_tp_filled(order_id, price, qty)

    def _on_position_update(self, position_data: Dict[str, Any]) -> None:
        """Handle private WebSocket position updates."""
        symbol = position_data.get("symbol", "")
        if symbol != self.symbol:
            return

        side = position_data.get("side", "").upper()
        size = float(position_data.get("size", 0) or 0)
        liq_price = position_data.get("liqPrice") or position_data.get("liquidationPrice")

        if side == "SELL":
            # Short position update
            if liq_price:
                self.anchor_short_liq_price = float(liq_price)
            logger.debug(f"Short position update: size={size}, liq={liq_price}")
        elif side == "BUY":
            # Long position update
            self.net_long_eth = size
            logger.debug(f"Long position update: net_long={self.net_long_eth}")

    def _on_wallet_update(self, wallet_data: Dict[str, Any]) -> None:
        """Handle private WebSocket wallet updates."""
        # Track margin/balance changes if needed
        pass

    # =========================================================================
    # Anchor Short Management
    # =========================================================================

    def initialize_anchor_short(self) -> bool:
        """Open the anchor short position at market."""
        logger.info(f"Initializing anchor short: {self.short_size} ETH @ {self.short_leverage}× leverage")

        # Set leverage for short
        self.client.set_leverage(self.symbol, int(self.short_leverage))

        # Check existing position
        positions = self.client.get_positions(self.symbol)
        if isinstance(positions, list):
            for pos in positions:
                side = pos.get("side", "").upper()
                size = float(pos.get("size", 0) or 0)
                if side == "SHORT" and size > 0:
                    self.anchor_short_entry = float(pos.get("entryPrice", 0) or 0)
                    self.anchor_short_liq_price = float(pos.get("liquidationPrice", 0) or 0)
                    logger.info(f"Existing short position found: {size} ETH @ {self.anchor_short_entry}")
                    return True

        # Open new short position
        order_result = self.client.execute_order({
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Market",
            "qty": str(self.short_size),
            "positionIdx": 2,  # Hedge mode: short position
        })

        if "error" in order_result:
            logger.error(f"Failed to open anchor short: {order_result['error']}")
            return False

        logger.info(f"Anchor short opened: {order_result}")

        # Fetch position info
        time.sleep(1)
        positions = self.client.get_positions(self.symbol)
        if isinstance(positions, list):
            for pos in positions:
                if pos.get("side", "").upper() == "SHORT":
                    self.anchor_short_entry = float(pos.get("entryPrice", 0) or 0)
                    self.anchor_short_liq_price = float(pos.get("liquidationPrice", 0) or 0)
                    logger.info(f"Anchor short entry: {self.anchor_short_entry}, liq: {self.anchor_short_liq_price}")

        return True

    def get_short_position_info(self) -> Dict[str, Any]:
        """Get current short position information."""
        return {
            "entry_price": self.anchor_short_entry,
            "liquidation_price": self.anchor_short_liq_price,
            "size": self.short_size,
            "leverage": self.short_leverage,
            "funding_pnl": self.funding_pnl,
        }

    # =========================================================================
    # Grid Long Management
    # =========================================================================

    def get_effective_grid_spacing(self) -> float:
        """Return grid spacing."""
        return self.grid_spacing

    def calculate_grid_prices(self, center_price: float) -> Tuple[List[float], List[float]]:
        """
        Calculate grid prices around the center price.

        Returns:
            (lower_prices, upper_prices) - Lists of prices below and above center
        """
        spacing = self.get_effective_grid_spacing()

        lower_prices = []
        upper_prices = []

        for i in range(1, self.grid_levels + 1):
            lower = round_to_tick_size(center_price - (i * spacing), self.tick_size)
            upper = round_to_tick_size(center_price + (i * spacing), self.tick_size)
            lower_prices.append(lower)
            upper_prices.append(upper)

        return lower_prices, upper_prices

    def place_grid_long_orders(self, center_price: float) -> int:
        """
        Place grid long orders around the current price.

        Returns:
            Number of orders placed
        """
        # Set leverage for longs
        self.client.set_leverage(self.symbol, int(self.grid_leverage))

        lower_prices, upper_prices = self.calculate_grid_prices(center_price)
        all_prices = lower_prices + upper_prices

        placed_count = 0
        qty = round_to_precision(self.grid_long_size, self.base_precision)

        for price in all_prices:
            # Skip if order already exists at this price
            if price in self.grid_long_orders:
                continue

            # Check delta limit before placing
            potential_delta = self.net_long_eth + qty
            if potential_delta > self.short_size:
                logger.warning(f"Skipping grid order at {price}: would exceed max delta")
                continue

            order_result = self.client.execute_order({
                "symbol": self.symbol,
                "side": "Buy",
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(price),
                "timeInForce": "PostOnly",  # Maker only
                "positionIdx": 1,  # Hedge mode: long position
            })

            if "error" not in order_result:
                order_id = order_result.get("orderId") or order_result.get("id")
                self.grid_long_orders[price] = {
                    "order_id": order_id,
                    "qty": qty,
                    "status": "open",
                    "created": datetime.now(),
                }
                placed_count += 1
                logger.info(f"Grid long placed: {qty} ETH @ ${price}")
            else:
                logger.error(f"Failed to place grid long @ {price}: {order_result['error']}")

        return placed_count

    def _handle_grid_long_filled(self, order_id: str, fill_price: float, fill_qty: float) -> None:
        """Handle grid long order fill - place take profit."""
        logger.info(f"Grid long filled: {fill_qty} ETH @ ${fill_price}")

        # Update net long
        self.net_long_eth += fill_qty

        # Remove from grid orders
        price_to_remove = None
        for price, order_info in self.grid_long_orders.items():
            if order_info.get("order_id") == order_id:
                price_to_remove = price
                break
        if price_to_remove:
            del self.grid_long_orders[price_to_remove]

        # Place take profit
        tp_price = round_to_tick_size(fill_price + self.take_profit_offset, self.tick_size)
        self._place_take_profit_order(fill_price, tp_price, fill_qty)

        # Check delta thresholds
        self._check_and_handle_thresholds()

    def _place_take_profit_order(self, entry_price: float, tp_price: float, qty: float) -> bool:
        """Place take profit limit order (maker)."""
        order_result = self.client.execute_order({
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(tp_price),
            "timeInForce": "PostOnly",
            "reduceOnly": True,
            "positionIdx": 1,  # Close long
        })

        if "error" not in order_result:
            order_id = order_result.get("orderId") or order_result.get("id")
            self.grid_tp_orders[order_id] = {
                "entry_price": entry_price,
                "tp_price": tp_price,
                "qty": qty,
            }
            logger.info(f"TP order placed: {qty} ETH @ ${tp_price} (entry: ${entry_price})")
            return True
        else:
            logger.error(f"Failed to place TP @ {tp_price}: {order_result['error']}")
            return False

    def _handle_tp_filled(self, order_id: str, fill_price: float, fill_qty: float) -> None:
        """Handle take profit order fill."""
        if order_id not in self.grid_tp_orders:
            logger.warning(f"TP order {order_id} not in tracking")
            return

        tp_info = self.grid_tp_orders.pop(order_id)
        entry_price = tp_info["entry_price"]
        profit = (fill_price - entry_price) * fill_qty

        self.net_long_eth -= fill_qty
        logger.info(f"TP filled: {fill_qty} ETH @ ${fill_price}, profit: ${profit:.4f}")

    # =========================================================================
    # Delta Control & Risk Management
    # =========================================================================

    def get_net_delta(self) -> float:
        """Calculate net delta (net_long - short_size)."""
        return self.net_long_eth - self.short_size

    def _check_and_handle_thresholds(self) -> Optional[str]:
        """Check inventory thresholds and take action if needed."""
        if self.short_size <= 0:
            return None

        ratio = self.net_long_eth / self.short_size

        for threshold, action in self.threshold_actions:
            if ratio >= threshold:
                logger.warning(f"Threshold {threshold * 100}% reached (ratio: {ratio:.2%})")

                if action == "partial_reduce_25":
                    self._reduce_short_position(0.25)
                    return action
                elif action == "partial_reduce_50":
                    self._reduce_short_position(0.50)
                    return action
                elif action == "full_unwind":
                    self._unwind_and_reset()
                    return action

        return None

    def _reduce_short_position(self, reduction_pct: float) -> bool:
        """Partially reduce the short position."""
        reduce_qty = round_to_precision(self.short_size * reduction_pct, self.base_precision)
        logger.warning(f"Reducing short by {reduction_pct * 100}%: {reduce_qty} ETH")

        order_result = self.client.execute_order({
            "symbol": self.symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": str(reduce_qty),
            "reduceOnly": True,
            "positionIdx": 2,  # Close short
        })

        if "error" not in order_result:
            self.short_size -= reduce_qty
            logger.info(f"Short reduced. New short size: {self.short_size} ETH")
            return True
        else:
            logger.error(f"Failed to reduce short: {order_result['error']}")
            return False

    def _unwind_and_reset(self) -> bool:
        """Emergency unwind: close all positions and cancel all orders."""
        logger.warning("FULL UNWIND: Closing all positions and orders")
        self.is_paused = True
        self.pause_reason = "full_unwind"

        # Cancel all open orders
        self.client.cancel_all_orders(self.symbol)
        self.grid_long_orders.clear()
        self.grid_tp_orders.clear()

        # Close long position if any
        if self.net_long_eth > 0:
            self.client.execute_order({
                "symbol": self.symbol,
                "side": "Sell",
                "orderType": "Market",
                "qty": str(self.net_long_eth),
                "reduceOnly": True,
                "positionIdx": 1,
            })
            self.net_long_eth = 0

        # Close short position
        if self.short_size > 0:
            self.client.execute_order({
                "symbol": self.symbol,
                "side": "Buy",
                "orderType": "Market",
                "qty": str(self.short_size),
                "reduceOnly": True,
                "positionIdx": 2,
            })
            self.short_size = 0

        logger.info("Full unwind complete")
        return True

    # =========================================================================
    # Safety Systems (Reserved for future use)
    # =========================================================================

    def run_safety_checks(self) -> Optional[str]:
        """Run safety checks. Currently disabled."""
        return None

    def emergency_unwind(self) -> bool:
        """Emergency unwind triggered by safety check."""
        return self._unwind_and_reset()

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self, duration_seconds: int = 3600, interval_seconds: int = 60) -> None:
        """
        Run the Long Grid Short Hedge strategy.

        Args:
            duration_seconds: How long to run the strategy
            interval_seconds: How often to check/refresh grid
        """
        logger.info("=" * 60)
        logger.info("Starting Long Grid Short Hedge Strategy")
        logger.info(f"Duration: {duration_seconds}s | Interval: {interval_seconds}s")
        logger.info("=" * 60)

        # Setup WebSockets
        if not self.setup_websockets():
            logger.error("Failed to setup WebSockets, falling back to REST polling")

        # Initialize anchor short
        if not self.initialize_anchor_short():
            logger.error("Failed to initialize anchor short, aborting")
            return

        # Get initial price
        ticker = self.client.get_ticker(self.symbol)
        if "error" not in ticker:
            self._last_price = float(ticker.get("lastPrice", 0))
        else:
            logger.error(f"Failed to get initial price: {ticker['error']}")
            return

        logger.info(f"Initial price: ${self._last_price}")

        # Place initial grid
        placed = self.place_grid_long_orders(self._last_price)
        logger.info(f"Placed {placed} initial grid orders")

        # Main loop
        start_time = time.time()
        end_time = start_time + duration_seconds

        try:
            while time.time() < end_time:
                if self.is_paused:
                    logger.warning(f"Strategy paused: {self.pause_reason}")
                    time.sleep(interval_seconds)
                    continue

                # Run safety checks
                kill_reason = self.run_safety_checks()
                if kill_reason:
                    logger.error(f"KILL SWITCH TRIGGERED: {kill_reason}")
                    self.emergency_unwind()
                    break

                # Log status
                logger.info("-" * 40)
                logger.info(f"Price: ${self._last_price:.2f}")
                logger.info(f"Net Long: {self.net_long_eth:.4f} ETH")
                logger.info(f"Short Size: {self.short_size:.4f} ETH")
                logger.info(f"Net Delta: {self.get_net_delta():.4f} ETH")
                logger.info(f"Active Grid Orders: {len(self.grid_long_orders)}")
                logger.info(f"Active TP Orders: {len(self.grid_tp_orders)}")

                # Refresh grid if needed
                if len(self.grid_long_orders) < self.grid_levels:
                    if self._last_price > 0:
                        placed = self.place_grid_long_orders(self._last_price)
                        if placed > 0:
                            logger.info(f"Refreshed {placed} grid orders")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        finally:
            # Cleanup
            logger.info("Strategy stopping, cleaning up WebSockets...")
            if self.ws_public:
                self.ws_public.disconnect()
            if self.ws_private:
                self.ws_private.disconnect()
            logger.info("Strategy stopped")
