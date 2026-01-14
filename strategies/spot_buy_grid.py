"""
Spot Buy Grid Strategy for Zoomex

A spot-only grid trading strategy designed for accumulating base assets.

Strategy Overview:
- Lower Band: Post-only limit buy orders below current price
- Upper Band: Conditional (trigger) + post-only buy orders above current price
- Take Profit: Limit sell orders placed when buy orders fill

This strategy is designed for Zoomex spot market only.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from logger import setup_logger
from strategies.market_maker import MarketMaker
from utils.helpers import round_to_precision, round_to_tick_size

logger = setup_logger("spot_buy_grid")


class SpotBuyGrid(MarketMaker):
    """
    Spot Buy Grid Strategy for Zoomex
    
    Features:
    - Lower Band: Post-only limit buy orders at grid intervals below current price
    - Upper Band: Conditional (trigger-based) + post-only buy orders above current price
    - Take Profit: Automatic limit sell orders when buys fill
    - Maximum orders per band limit
    
    Grid Example (current price = $1.00, grid_spacing = $0.02):
    
        Upper Band (Conditional Buys):
        $1.10 - Conditional buy, triggers at $1.10, limit at $1.10 - trigger_offset
        $1.08 - Conditional buy, triggers at $1.08
        $1.06 - Conditional buy, triggers at $1.06
        $1.04 - Conditional buy, triggers at $1.04
        $1.02 - Conditional buy, triggers at $1.02
        
        $1.00 === CURRENT PRICE ===
        
        Lower Band (Post-Only Limit Buys):
        $0.98 - Post-only limit buy
        $0.96 - Post-only limit buy
        $0.94 - Post-only limit buy
        $0.92 - Post-only limit buy
        $0.90 - Post-only limit buy
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbol: str,
        grid_spacing: float,                        # Price spacing between grid levels
        maximum_order: int = 5,                     # Max orders per band
        order_quantity: Optional[float] = None,     # Size per order
        trigger_offset: float = 0.0,                # Upper band: offset from trigger to limit
        take_profit_offset: float = 0.0,            # Take profit offset from buy price
        exchange: str = 'zoomex',
        exchange_config: Optional[Dict[str, Any]] = None,
        enable_database: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize Spot Buy Grid Strategy.
        
        Args:
            api_key: API key
            secret_key: API secret
            symbol: Trading pair (e.g., "MNTUSDT")
            grid_spacing: Price interval between grid levels
            maximum_order: Maximum orders per band (upper and lower)
            order_quantity: Size per order in base asset
            trigger_offset: For upper band, offset between trigger and limit price
            take_profit_offset: Profit offset for sell orders
            exchange: Exchange name (must be 'zoomex')
            exchange_config: Exchange configuration
            enable_database: Whether to enable database logging
        """
        # Force exchange to zoomex and set spot category
        if exchange != 'zoomex':
            raise ValueError("SpotBuyGrid strategy only supports Zoomex exchange")
        
        # Update exchange config for spot trading
        if exchange_config is None:
            exchange_config = {}
        exchange_config['category'] = 'spot'
        
        super().__init__(
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            base_spread_percentage=0.1,  # Not used in grid strategy
            order_quantity=order_quantity,
            max_orders=maximum_order,
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # Grid parameters
        self.grid_spacing = grid_spacing
        self.maximum_order = maximum_order
        self.trigger_offset = trigger_offset
        self.take_profit_offset = take_profit_offset

        # Grid state tracking
        self.grid_initialized = False
        self.reference_price: Optional[float] = None
        
        # Lower band orders: {grid_price: order_info}
        self.lower_band_orders: Dict[float, Dict[str, Any]] = {}
        
        # Upper band orders (conditional): {grid_price: order_info}
        self.upper_band_orders: Dict[float, Dict[str, Any]] = {}
        
        # Take profit orders: {order_id: {buy_price, quantity, ...}}
        self.take_profit_orders: Dict[str, Dict[str, Any]] = {}
        
        # Filled buys awaiting take profit
        self.pending_take_profits: List[Dict[str, Any]] = []
        
        # Order ID mappings
        self.order_id_to_grid_price: Dict[str, float] = {}
        self.order_id_to_band: Dict[str, str] = {}  # 'lower', 'upper', 'tp'
        
        # Statistics
        self.total_buys_filled = 0
        self.total_sells_filled = 0
        self.total_profit = 0.0

        logger.info("=" * 60)
        logger.info("Spot Buy Grid Strategy Initialized")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Grid Spacing: {grid_spacing}")
        logger.info(f"Maximum Orders per Band: {maximum_order}")
        logger.info(f"Order Quantity: {order_quantity}")
        logger.info(f"Trigger Offset: {trigger_offset}")
        logger.info(f"Take Profit Offset: {take_profit_offset}")
        logger.info("=" * 60)

    # ==================== Balance Methods ====================

    def get_balance(self) -> Dict[str, Dict]:
        """Get spot account balances."""
        return self.client.get_spot_balance()

    def get_ticker(self) -> Dict:
        """Get spot ticker for the symbol."""
        return self.client.get_spot_ticker(self.symbol)

    # ==================== Grid Initialization ====================

    def _calculate_grid_levels(self, reference_price: float) -> tuple:
        """
        Calculate grid price levels for both bands.
        
        Returns:
            (lower_prices, upper_prices) - Lists of prices for each band
        """
        lower_prices = []
        upper_prices = []
        
        # Lower band: prices below reference
        for i in range(1, self.maximum_order + 1):
            price = reference_price - (i * self.grid_spacing)
            price = round_to_tick_size(price, self.tick_size)
            if price > 0:
                lower_prices.append(price)
        
        # Upper band: prices above reference
        for i in range(1, self.maximum_order + 1):
            price = reference_price + (i * self.grid_spacing)
            price = round_to_tick_size(price, self.tick_size)
            upper_prices.append(price)
        
        return lower_prices, upper_prices

    def initialize_grid(self) -> bool:
        """
        Initialize the grid with orders on both bands.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing spot buy grid...")
        
        # Get current price
        ticker = self.get_ticker()
        if "error" in ticker:
            logger.error(f"Failed to get ticker: {ticker['error']}")
            return False
        
        current_price = float(ticker.get('lastPrice', 0))
        if current_price <= 0:
            logger.error("Invalid current price")
            return False
        
        self.reference_price = current_price
        logger.info(f"Reference price: {current_price}")
        
        # Get symbol info for precision
        symbol_info = self.client.get_spot_symbol_info(self.symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {self.symbol}")
            return False
            
        self.tick_size = symbol_info.get('tick_size', 0.01)
        self.qty_step = symbol_info.get('qty_step', 0.0001)
        self.quote_asset = symbol_info.get('quote_asset', 'USDC')
        
        logger.info(f"Symbol Info: tick_size={self.tick_size}, qty_step={self.qty_step}")
        
        # Calculate grid levels
        lower_prices, upper_prices = self._calculate_grid_levels(current_price)
        
        logger.info(f"Lower band prices: {lower_prices}")
        logger.info(f"Upper band prices: {upper_prices}")
        
        # Cancel any existing orders
        self._cancel_all_orders()
        
        # Place lower band orders (post-only limit buys)
        placed_lower = 0
        for price in lower_prices:
            if self._place_lower_band_order(price):
                placed_lower += 1
        
        logger.info(f"Placed {placed_lower} lower band orders")
        
        # Place upper band orders (conditional + post-only buys)
        placed_upper = 0
        for trigger_price in upper_prices:
            if self._place_upper_band_order(trigger_price):
                placed_upper += 1
        
        logger.info(f"Placed {placed_upper} upper band orders")
        
        self.grid_initialized = True
        return True

    # ==================== Order Placement ====================

    def _place_lower_band_order(self, price: float) -> bool:
        """
        Place a post-only limit buy order in the lower band.
        
        Args:
            price: Limit price for the buy order
            
        Returns:
            True if order placed successfully
        """
        quantity = round_to_tick_size(self.order_quantity, self.qty_step)
        
        order_details = {
            'symbol': self.symbol,
            'side': 'Buy',
            'orderType': 'Limit',
            'qty': quantity,
            'price': price,
            'timeInForce': 'PostOnly',
            'orderFilter': 'Order',
        }
        
        logger.info(f"Placing lower band buy: price={price}, qty={quantity}")
        response = self.client.execute_spot_order(order_details)
        
        if "error" in response:
            logger.error(f"Failed to place lower band order: {response['error']}")
            return False
        
        order_id = response.get('orderId')
        if order_id:
            self.lower_band_orders[price] = {
                'order_id': order_id,
                'price': price,
                'quantity': quantity,
                'status': 'open',
                'created_time': datetime.now(),
            }
            self.order_id_to_grid_price[order_id] = price
            self.order_id_to_band[order_id] = 'lower'
            logger.info(f"Lower band order placed: ID={order_id}, price={price}")
            return True
        
        return False

    def _place_upper_band_order(self, trigger_price: float) -> bool:
        """
        Place a conditional + post-only buy order in the upper band.
        
        Trigger activates when price rises to trigger_price.
        Limit price is trigger_price - trigger_offset.
        
        Args:
            trigger_price: Price that activates the order
            
        Returns:
            True if order placed successfully
        """
        quantity = round_to_tick_size(self.order_quantity, self.qty_step)
        
        # Limit price is slightly below trigger to ensure fill as maker
        limit_price = round_to_tick_size(
            trigger_price - self.trigger_offset, 
            self.tick_size
        )
        
        order_details = {
            'symbol': self.symbol,
            'side': 'Buy',
            'orderType': 'Limit',
            'qty': quantity,
            'price': limit_price,
            'triggerPrice': trigger_price,
            'triggerDirection': 1,  # Trigger when price rises to
            'timeInForce': 'PostOnly',
            'orderFilter': 'StopOrder',
        }
        
        logger.info(f"Placing upper band conditional buy: trigger={trigger_price}, limit={limit_price}, qty={quantity}")
        response = self.client.execute_spot_order(order_details)
        
        if "error" in response:
            logger.error(f"Failed to place upper band order: {response['error']}")
            return False
        
        order_id = response.get('orderId')
        if order_id:
            self.upper_band_orders[trigger_price] = {
                'order_id': order_id,
                'trigger_price': trigger_price,
                'limit_price': limit_price,
                'quantity': quantity,
                'status': 'open',
                'created_time': datetime.now(),
            }
            self.order_id_to_grid_price[order_id] = trigger_price
            self.order_id_to_band[order_id] = 'upper'
            logger.info(f"Upper band order placed: ID={order_id}, trigger={trigger_price}")
            return True
        
        return False

    def _place_take_profit_order(self, buy_price: float, quantity: float) -> bool:
        """
        Place a limit sell order for take profit.
        
        Args:
            buy_price: Price at which the buy was filled
            quantity: Quantity to sell
            
        Returns:
            True if order placed successfully
        """
        tp_price = round_to_tick_size(
            buy_price + self.take_profit_offset,
            self.tick_size
        )
        quantity = round_to_precision(quantity, self.qty_step)
        
        order_details = {
            'symbol': self.symbol,
            'side': 'Sell',
            'orderType': 'Limit',
            'qty': quantity,
            'price': tp_price,
            'timeInForce': 'PostOnly',
            'orderFilter': 'Order',
        }
        
        logger.info(f"Placing take profit sell: price={tp_price}, qty={quantity}, buy_price={buy_price}")
        response = self.client.execute_spot_order(order_details)
        
        if "error" in response:
            logger.error(f"Failed to place take profit order: {response['error']}")
            return False
        
        order_id = response.get('orderId')
        if order_id:
            self.take_profit_orders[order_id] = {
                'order_id': order_id,
                'buy_price': buy_price,
                'sell_price': tp_price,
                'quantity': quantity,
                'status': 'open',
                'created_time': datetime.now(),
            }
            self.order_id_to_band[order_id] = 'tp'
            logger.info(f"Take profit order placed: ID={order_id}, price={tp_price}")
            return True
        
        return False

    # ==================== Order Management ====================

    def _cancel_all_orders(self) -> None:
        """Cancel all open orders (regular and conditional)."""
        logger.info("Cancelling all open orders...")
        
        # Cancel regular orders
        result = self.client.cancel_all_spot_orders(self.symbol, order_filter="Order")
        if "error" in result:
            logger.warning(f"Error cancelling regular orders: {result['error']}")
        
        # Cancel conditional orders
        result = self.client.cancel_all_spot_orders(self.symbol, order_filter="StopOrder")
        if "error" in result:
            logger.warning(f"Error cancelling conditional orders: {result['error']}")
        
        # Clear local tracking
        self.lower_band_orders.clear()
        self.upper_band_orders.clear()
        self.take_profit_orders.clear()
        self.order_id_to_grid_price.clear()
        self.order_id_to_band.clear()

    def _sync_orders_with_exchange(self) -> None:
        """
        Synchronize local order tracking with exchange state.
        Detect filled orders and place corresponding actions.
        """
        # Get regular open orders
        regular_orders = self.client.get_spot_open_orders(self.symbol, order_filter="Order")
        if isinstance(regular_orders, dict) and "error" in regular_orders:
            logger.error(f"Failed to get regular orders: {regular_orders['error']}")
            return
        
        # Get conditional open orders
        conditional_orders = self.client.get_spot_open_orders(self.symbol, order_filter="StopOrder")
        if isinstance(conditional_orders, dict) and "error" in conditional_orders:
            logger.error(f"Failed to get conditional orders: {conditional_orders['error']}")
            conditional_orders = []
        
        # Combine all open order IDs and Prices
        open_order_ids = set()
        self.active_order_prices = set()
        
        for order in regular_orders:
            if order.get('orderId'):
                open_order_ids.add(order['orderId'])
            if order.get('price'):
                try:
                    p = float(order['price'])
                    self.active_order_prices.add(p)
                except:
                    pass
                    
        for order in conditional_orders:
            if order.get('orderId'):
                open_order_ids.add(order['orderId'])
            # Track conditional order trigger prices
            if order.get('triggerPrice'):
                try:
                    p = float(order['triggerPrice'])
                    self.active_order_prices.add(p)
                except:
                    pass
        
        # Check lower band orders for fills
        # ... (unchanged)

    # ... (handlers)

    def _refill_grid_orders(self) -> None:
        """
        Refill missing grid orders up to maximum_order per band.
        """
        if not self.reference_price:
            return
        
        # Use the fixed reference price for grid stability
        lower_prices, upper_prices = self._calculate_grid_levels(self.reference_price)
        
        # Count current orders
        lower_count = len(self.lower_band_orders)
        upper_count = len(self.upper_band_orders)
        
        # Refill lower band
        if lower_count < self.maximum_order:
            for price in lower_prices:
                if price not in self.lower_band_orders and lower_count < self.maximum_order:
                    # Check if ANY order exists at this price on exchange
                    if price in self.active_order_prices:
                        logger.debug(f"Skipping refill at {price}, order already on exchange")
                        lower_count += 1  # Count this existing order towards the limit
                        continue
                        
                    if self._place_lower_band_order(price):
                        lower_count += 1
        
        # Refill upper band
        if upper_count < self.maximum_order:
            for trigger_price in upper_prices:
                if trigger_price not in self.upper_band_orders and upper_count < self.maximum_order:
                    # Check if ANY conditional order exists at this trigger price
                    if trigger_price in self.active_order_prices:
                        logger.debug(f"Skipping upper refill at {trigger_price}, order already on exchange")
                        upper_count += 1  # Count this existing order towards the limit
                        continue
                        
                    if self._place_upper_band_order(trigger_price):
                        upper_count += 1
        for price, order_info in list(self.lower_band_orders.items()):
            order_id = order_info.get('order_id')
            if order_id and order_id not in open_order_ids:
                # Order missing from open list.
                # It could be filled, canceled, or pagination missed it.
                # If we assume it's gone, we remove it.
                # Refill will try to replace it.
                # But Refill will now check active_order_prices first!
                logger.info(f"Lower band order finished (filled/canceled): price={price}, ID={order_id}")
                
                # Check if it was filled (simple heuristic for now: if not in open, assume filled if we don't have fill history check)
                # Ideally check fill history here.
                # For now, we behave as before but safer refill.
                self._handle_buy_fill(price, order_info['quantity'], 'lower')
                del self.lower_band_orders[price]
        
        # ... (Upper band logic similar) ...
        # Check upper band orders for fills
        for trigger_price, order_info in list(self.upper_band_orders.items()):
            order_id = order_info.get('order_id')
            if order_id and order_id not in open_order_ids:
                fill_price = order_info.get('limit_price', trigger_price)
                logger.info(f"Upper band order finished: trigger={trigger_price}, ID={order_id}")
                self._handle_buy_fill(fill_price, order_info['quantity'], 'upper')
                del self.upper_band_orders[trigger_price]
        
        # Check take profit orders for fills
        for order_id, order_info in list(self.take_profit_orders.items()):
            if order_id not in open_order_ids:
                logger.info(f"Take profit order finished: ID={order_id}")
                self._handle_tp_fill(order_info)
                del self.take_profit_orders[order_id]

    # ... (handlers) ...

    def _refill_grid_orders(self) -> None:
        """
        Refill missing grid orders up to maximum_order per band.
        """
        if not self.reference_price:
            return
        
        # Use the fixed reference price for grid stability
        lower_prices, upper_prices = self._calculate_grid_levels(self.reference_price)
        
        # Count current orders
        lower_count = len(self.lower_band_orders)
        upper_count = len(self.upper_band_orders)
        
        # Refill lower band
        if lower_count < self.maximum_order:
            for price in lower_prices:
                if price not in self.lower_band_orders and lower_count < self.maximum_order:
                    # Check if ANY order exists at this price on exchange
                    if price in self.active_order_prices:
                        logger.debug(f"Skipping refill at {price}, order already on exchange")
                        continue
                        
                    if self._place_lower_band_order(price):
                        lower_count += 1
        
        # Refill upper band
        if upper_count < self.maximum_order:
            for trigger_price in upper_prices:
                if trigger_price not in self.upper_band_orders and upper_count < self.maximum_order:
                    # Note: conditional orders triggers are not in 'price' field of regular orders
                    if self._place_upper_band_order(trigger_price):
                        upper_count += 1

    # ==================== Main Loop ====================

    def place_limit_orders(self) -> None:
        """Override parent method - called during main loop."""
        # Sync with exchange and detect fills
        self._sync_orders_with_exchange()
        
        # Refill grid orders
        self._refill_grid_orders()

    def calculate_prices(self) -> List[tuple]:
        """Not used in grid strategy."""
        return []

    def need_rebalance(self) -> bool:
        """Grid strategy doesn't use rebalancing."""
        return False

    def rebalance_position(self) -> None:
        """Grid strategy doesn't use rebalancing."""
        pass

    def _get_extra_summary_sections(self) -> str:
        """Add grid-specific stats to summary."""
        lines = [
            "\n" + "=" * 40,
            "SPOT BUY GRID STATS",
            "=" * 40,
            f"Reference Price: {self.reference_price or 'N/A'}",
            f"Grid Spacing: {self.grid_spacing}",
            f"Lower Band Orders: {len(self.lower_band_orders)}",
            f"Upper Band Orders: {len(self.upper_band_orders)}",
            f"Take Profit Orders: {len(self.take_profit_orders)}",
            "-" * 40,
            f"Total Buys Filled: {self.total_buys_filled}",
            f"Total Sells Filled: {self.total_sells_filled}",
            f"Total Profit: {self.total_profit:.4f} {self.quote_asset}",
        ]
        return "\n".join(lines)

    def run(self, duration_seconds: int = 3600, interval_seconds: int = 60) -> None:
        """
        Run the Spot Buy Grid strategy.
        
        Args:
            duration_seconds: Total runtime in seconds
            interval_seconds: Interval between updates
        """
        logger.info("=" * 60)
        logger.info("Starting Spot Buy Grid Strategy")
        logger.info("=" * 60)
        
        # Initialize grid
        if not self.initialize_grid():
            logger.error("Failed to initialize grid, aborting")
            return
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        try:
            while time.time() < end_time and not getattr(self, '_stop_flag', False):
                loop_start = time.time()
                
                try:
                    # Main update cycle
                    self.place_limit_orders()
                    
                    # Log status
                    logger.info(
                        f"Grid Status: Lower={len(self.lower_band_orders)}, "
                        f"Upper={len(self.upper_band_orders)}, "
                        f"TP={len(self.take_profit_orders)}, "
                        f"Buys={self.total_buys_filled}, "
                        f"Profits={self.total_profit:.4f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Calculate sleep time
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval_seconds - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            logger.info("Cleaning up...")
            self._cancel_all_orders()
            logger.info(self._get_extra_summary_sections())
            logger.info("Spot Buy Grid Strategy stopped")
