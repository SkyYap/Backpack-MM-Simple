"""
Spot Long Grid Strategy

Dual-mode grid trading strategy for spot markets on Zoomex.

Core Concept:
- Lower Grid: PostOnly limit buys (catch dips)
- Upper Grid: Conditional + PostOnly (catch breakouts as maker)
- Take profit sells when buys fill
- All orders are maker-only
- Maintains exactly max_orders_per_side on each side

Author: Antigravity
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from logger import setup_logger
from strategies.market_maker import MarketMaker
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("spot_long_grid")


class SpotLongGrid(MarketMaker):
    """
    Dual-mode spot grid strategy for Zoomex.
    
    Strategy Overview:
    - Lower Grid: Limit buy orders below current price (PostOnly)
    - Upper Grid: Conditional buy orders above current price (Trigger + PostOnly)
    - Take Profit: Sell orders placed when buys fill
    - All orders are maker-only to minimize fees
    - Maintains exactly max_orders_per_side orders on each side
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str = "ETHUSDC",
        # === Grid Parameters ===
        grid_size: float = 0.02,          # Buy size per level (ETH)
        grid_spacing: float = 1.0,        # USD between levels
        max_orders_per_side: int = 5,     # Max active orders per side
        take_profit: float = 1.5,         # TP offset (USD)
        trigger_offset: float = 0.5,      # Limit below trigger for upper grid
        # === Exchange Config ===
        exchange: str = "zoomex",
        exchange_config: Optional[Dict[str, Any]] = None,
        enable_database: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize Spot Long Grid Strategy.

        Args:
            symbol: Spot trading pair (e.g., "ETHUSDC")
            grid_size: Order size per grid level
            grid_spacing: USD distance between grid levels
            max_orders_per_side: Maximum active orders per side (default: 5)
            take_profit: USD profit target per trade
            trigger_offset: For upper grid, limit price = trigger - offset
        """
        # Force Zoomex exchange
        if exchange != "zoomex":
            logger.warning("SpotLongGrid only supports Zoomex, forcing exchange='zoomex'")
            exchange = "zoomex"

        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # === Grid Parameters ===
        self.grid_size = abs(grid_size)
        self.grid_spacing = abs(grid_spacing)
        self.max_orders_per_side = max(1, max_orders_per_side)
        self.take_profit_offset = abs(take_profit)
        self.trigger_offset = abs(trigger_offset)

        # === Order Tracking ===
        # Lower grid: {price: {order_id, qty, status}}
        self.lower_grid_orders: Dict[float, Dict[str, Any]] = {}
        # Upper grid (conditional): {trigger_price: {order_id, limit_price, qty}}
        self.upper_grid_orders: Dict[float, Dict[str, Any]] = {}
        # Take profit orders: {order_id: {entry_price, qty}}
        self.tp_orders: Dict[str, Dict[str, Any]] = {}

        # === State ===
        self._last_price: float = 0.0
        self._center_price: float = 0.0
        self.is_running: bool = False

        logger.info("=" * 60)
        logger.info("Spot Long Grid Strategy Initialized")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Grid Size: {self.grid_size} per order")
        logger.info(f"Grid Spacing: ${self.grid_spacing}")
        logger.info(f"Max Orders Per Side: {self.max_orders_per_side}")
        logger.info(f"Take Profit: ${self.take_profit_offset}")
        logger.info(f"Trigger Offset: ${self.trigger_offset}")
        logger.info("=" * 60)

    # =========================================================================
    # Grid Price Calculation
    # =========================================================================

    def calculate_grid_prices(self, center_price: float, levels: int) -> Tuple[List[float], List[float]]:
        """
        Calculate grid prices around the center price.

        Args:
            center_price: Current price to center grid around
            levels: Number of levels to calculate each side

        Returns:
            (lower_prices, upper_prices) - Lists of prices, sorted by distance from center
        """
        lower_prices = []
        upper_prices = []

        for i in range(1, levels + 1):
            lower = round_to_tick_size(center_price - (i * self.grid_spacing), self.tick_size)
            upper = round_to_tick_size(center_price + (i * self.grid_spacing), self.tick_size)
            lower_prices.append(lower)
            upper_prices.append(upper)

        return lower_prices, upper_prices

    # =========================================================================
    # Order Rebalancing - Keep N Nearest Orders
    # =========================================================================

    def rebalance_lower_grid(self, current_price: float) -> None:
        """
        Rebalance lower grid to maintain exactly max_orders_per_side orders.
        Cancel furthest orders if too many, add nearest if too few.
        """
        current_count = len(self.lower_grid_orders)
        
        # Calculate ideal prices (nearest to current price)
        ideal_prices, _ = self.calculate_grid_prices(current_price, self.max_orders_per_side)
        
        # If too many orders, cancel the furthest ones
        if current_count > self.max_orders_per_side:
            # Sort existing orders by distance from current price (furthest first)
            sorted_prices = sorted(self.lower_grid_orders.keys(), key=lambda p: current_price - p, reverse=True)
            
            orders_to_cancel = current_count - self.max_orders_per_side
            for i in range(orders_to_cancel):
                price = sorted_prices[i]
                order_info = self.lower_grid_orders[price]
                order_id = order_info.get("order_id")
                
                if order_id:
                    result = self.client.cancel_order(order_id, self.symbol)
                    if "error" not in result:
                        logger.info(f"Cancelled furthest lower grid @ ${price}")
                    else:
                        logger.warning(f"Failed to cancel order {order_id}: {result.get('error')}")
                
                del self.lower_grid_orders[price]
        
        # If too few orders, add more (nearest to current price)
        elif current_count < self.max_orders_per_side:
            orders_to_add = self.max_orders_per_side - current_count
            qty = round_to_precision(self.grid_size, self.base_precision)
            
            for price in ideal_prices:
                if orders_to_add <= 0:
                    break
                    
                if price in self.lower_grid_orders:
                    continue
                
                order_result = self.client.execute_order({
                    "symbol": self.symbol,
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": str(qty),
                    "price": str(price),
                    "timeInForce": "PostOnly",
                })

                if "error" not in order_result:
                    order_id = order_result.get("orderId") or order_result.get("id")
                    self.lower_grid_orders[price] = {
                        "order_id": order_id,
                        "qty": qty,
                        "status": "open",
                        "created": datetime.now(),
                    }
                    orders_to_add -= 1
                    logger.info(f"Added lower grid: BUY {qty} @ ${price}")
                else:
                    logger.error(f"Failed to place lower grid @ {price}: {order_result.get('error')}")

    def rebalance_upper_grid(self, current_price: float) -> None:
        """
        Rebalance upper grid to maintain exactly max_orders_per_side conditional orders.
        Cancel furthest orders if too many, add nearest if too few.
        """
        current_count = len(self.upper_grid_orders)
        
        # Calculate ideal trigger prices (nearest to current price)
        _, ideal_triggers = self.calculate_grid_prices(current_price, self.max_orders_per_side)
        
        # If too many orders, cancel the furthest ones
        if current_count > self.max_orders_per_side:
            # Sort existing orders by distance from current price (furthest first)
            sorted_triggers = sorted(self.upper_grid_orders.keys(), key=lambda p: p - current_price, reverse=True)
            
            orders_to_cancel = current_count - self.max_orders_per_side
            for i in range(orders_to_cancel):
                trigger = sorted_triggers[i]
                order_info = self.upper_grid_orders[trigger]
                order_id = order_info.get("order_id")
                
                if order_id:
                    result = self.client.cancel_order(order_id, self.symbol)
                    if "error" not in result:
                        logger.info(f"Cancelled furthest upper grid trigger @ ${trigger}")
                    else:
                        logger.warning(f"Failed to cancel order {order_id}: {result.get('error')}")
                
                del self.upper_grid_orders[trigger]
        
        # If too few orders, add more (nearest to current price)
        elif current_count < self.max_orders_per_side:
            orders_to_add = self.max_orders_per_side - current_count
            qty = round_to_precision(self.grid_size, self.base_precision)
            
            for trigger_price in ideal_triggers:
                if orders_to_add <= 0:
                    break
                    
                if trigger_price in self.upper_grid_orders:
                    continue
                
                limit_price = round_to_tick_size(trigger_price - self.trigger_offset, self.tick_size)
                
                order_result = self.client.execute_order({
                    "symbol": self.symbol,
                    "side": "Buy",
                    "orderType": "Limit",
                    "qty": str(qty),
                    "price": str(limit_price),
                    "triggerPrice": str(trigger_price),
                    "triggerDirection": 1,
                    "timeInForce": "PostOnly",
                })

                if "error" not in order_result:
                    order_id = order_result.get("orderId") or order_result.get("id")
                    self.upper_grid_orders[trigger_price] = {
                        "order_id": order_id,
                        "limit_price": limit_price,
                        "qty": qty,
                        "status": "pending",
                        "created": datetime.now(),
                    }
                    orders_to_add -= 1
                    logger.info(f"Added upper grid: trigger=${trigger_price} limit=${limit_price}")
                else:
                    logger.error(f"Failed to place upper grid @ {trigger_price}: {order_result.get('error')}")

    def rebalance_all_grids(self, current_price: float) -> None:
        """Rebalance both lower and upper grids around current price."""
        logger.debug(f"Rebalancing grids around ${current_price}")
        self.rebalance_lower_grid(current_price)
        self.rebalance_upper_grid(current_price)

    # =========================================================================
    # Order Fill Handlers
    # =========================================================================

    def _handle_buy_filled(self, price: float, qty: float, is_upper: bool = False) -> None:
        """Handle buy order fill - place take profit sell."""
        logger.info(f"Buy filled: {qty} @ ${price} ({'upper' if is_upper else 'lower'} grid)")

        # Remove from tracking
        if is_upper:
            trigger_to_remove = None
            for trigger, order_info in self.upper_grid_orders.items():
                if abs(order_info.get("limit_price", 0) - price) < 0.01:
                    trigger_to_remove = trigger
                    break
            if trigger_to_remove:
                del self.upper_grid_orders[trigger_to_remove]
        else:
            if price in self.lower_grid_orders:
                del self.lower_grid_orders[price]

        # Place take profit sell
        tp_price = round_to_tick_size(price + self.take_profit_offset, self.tick_size)
        self._place_take_profit_order(price, tp_price, qty)
        
        # Rebalance to fill the gap
        self.rebalance_all_grids(self._last_price)

    def _place_take_profit_order(self, entry_price: float, tp_price: float, qty: float) -> bool:
        """Place take profit limit sell order (PostOnly)."""
        order_result = self.client.execute_order({
            "symbol": self.symbol,
            "side": "Sell",
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(tp_price),
            "timeInForce": "PostOnly",
        })

        if "error" not in order_result:
            order_id = order_result.get("orderId") or order_result.get("id")
            self.tp_orders[order_id] = {
                "entry_price": entry_price,
                "tp_price": tp_price,
                "qty": qty,
            }
            logger.info(f"TP order placed: SELL {qty} @ ${tp_price} (entry: ${entry_price})")
            return True
        else:
            logger.error(f"Failed to place TP @ {tp_price}: {order_result.get('error')}")
            return False

    def _handle_tp_filled(self, order_id: str, fill_price: float, fill_qty: float) -> None:
        """Handle take profit fill - rebalance grid."""
        if order_id not in self.tp_orders:
            logger.warning(f"TP order {order_id} not in tracking")
            return

        tp_info = self.tp_orders.pop(order_id)
        entry_price = tp_info["entry_price"]
        profit = (fill_price - entry_price) * fill_qty

        logger.info(f"TP filled: SELL {fill_qty} @ ${fill_price}, profit: ${profit:.4f}")
        
        # Rebalance grids after TP fill
        self.rebalance_all_grids(self._last_price)

    # =========================================================================
    # Grid Management
    # =========================================================================

    def cancel_all_grid_orders(self) -> None:
        """Cancel all grid orders."""
        logger.info("Cancelling all grid orders...")
        self.client.cancel_all_orders(self.symbol)
        self.lower_grid_orders.clear()
        self.upper_grid_orders.clear()
        self.tp_orders.clear()
        logger.info("All grid orders cancelled")

    def get_grid_status(self) -> Dict[str, Any]:
        """Get current grid status."""
        return {
            "lower_grid_count": len(self.lower_grid_orders),
            "upper_grid_count": len(self.upper_grid_orders),
            "tp_orders_count": len(self.tp_orders),
            "max_per_side": self.max_orders_per_side,
            "last_price": self._last_price,
            "center_price": self._center_price,
        }

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self, duration_seconds: int = 3600, interval_seconds: int = 60) -> None:
        """
        Run the Spot Long Grid strategy.

        Args:
            duration_seconds: How long to run the strategy
            interval_seconds: How often to check/refresh grid
        """
        logger.info("=" * 60)
        logger.info("Starting Spot Long Grid Strategy")
        logger.info(f"Duration: {duration_seconds}s | Interval: {interval_seconds}s")
        logger.info(f"Max Orders Per Side: {self.max_orders_per_side}")
        logger.info("=" * 60)

        self.is_running = True

        # Get initial price
        ticker = self.client.get_ticker(self.symbol)
        if "error" not in ticker:
            self._last_price = float(ticker.get("lastPrice", 0))
            self._center_price = self._last_price
        else:
            logger.error(f"Failed to get initial price: {ticker.get('error')}")
            return

        logger.info(f"Initial price: ${self._last_price}")

        # Place initial grids (up to max_orders_per_side each)
        self.rebalance_all_grids(self._last_price)
        
        status = self.get_grid_status()
        logger.info(f"Initial grid: {status['lower_grid_count']} lower + {status['upper_grid_count']} upper")

        # Main loop
        start_time = time.time()
        end_time = start_time + duration_seconds

        try:
            while time.time() < end_time and self.is_running:
                # Update price
                ticker = self.client.get_ticker(self.symbol)
                if "error" not in ticker:
                    self._last_price = float(ticker.get("lastPrice", 0))

                # Log status
                status = self.get_grid_status()
                logger.info("-" * 40)
                logger.info(f"Price: ${self._last_price:.4f}")
                logger.info(f"Lower Grid: {status['lower_grid_count']}/{self.max_orders_per_side}")
                logger.info(f"Upper Grid: {status['upper_grid_count']}/{self.max_orders_per_side}")
                logger.info(f"TP Orders: {status['tp_orders_count']}")

                # Rebalance grids to maintain max_orders_per_side
                self.rebalance_all_grids(self._last_price)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")

        finally:
            logger.info("Strategy stopping...")
            self.is_running = False
            logger.info("Strategy stopped")

    def stop(self) -> None:
        """Stop the strategy."""
        self.is_running = False
        logger.info("Stop signal received")
