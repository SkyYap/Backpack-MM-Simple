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

import json
import math
import os
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
        max_orders_upper: int = 5,                  # Max orders for upper band
        max_orders_lower: int = 5,                  # Max orders for lower band
        order_quantity: Optional[float] = None,     # Size per order
        trigger_offset: float = 0.0,                # Upper band: offset from trigger to limit
        take_profit_offset: float = 0.0,            # Take profit offset from buy price
        trading_fee: float = 0.001,                 # Trading fee as decimal (0.001 = 0.1%)
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
            trading_fee: Trading fee as decimal (e.g., 0.001 = 0.1%), used to reduce TP quantity
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
            max_orders=max(max_orders_upper, max_orders_lower), # Pass raw max for parent compat (though not strictly used by parent logic in same way)
            exchange=exchange,
            exchange_config=exchange_config,
            enable_database=enable_database,
            **kwargs,
        )

        # Grid parameters
        self.grid_spacing = grid_spacing
        self.max_orders_upper = max_orders_upper
        self.max_orders_lower = max_orders_lower
        self.trigger_offset = trigger_offset
        self.take_profit_offset = take_profit_offset
        self.trading_fee = trading_fee  # Trading fee as decimal (0.001 = 0.1%)

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
        
        # Track filled price levels to prevent duplicate orders
        self.filled_lower_prices: Set[float] = set()
        self.filled_upper_prices: Set[float] = set()
        
        # Track processed fill IDs to avoid duplicate detection
        self._processed_fill_ids: Set[str] = set()
        
        # State persistence file path
        self._state_file = self._get_state_file_path()
        
        # Sell-only mode: stop all buying, keep only TP sells active
        self.sell_only_mode = False

        logger.info("=" * 60)
        logger.info("Spot Buy Grid Strategy Initialized")
        logger.info("=" * 60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Grid Spacing: {grid_spacing}")
        logger.info(f"Max Orders Upper: {max_orders_upper}")
        logger.info(f"Max Orders Lower: {max_orders_lower}")
        logger.info(f"Order Quantity: {order_quantity}")
        logger.info(f"Trigger Offset: {trigger_offset}")
        logger.info(f"Take Profit Offset: {take_profit_offset}")
        logger.info(f"Trading Fee: {trading_fee * 100:.2f}%")
        logger.info("=" * 60)

    def _get_state_file_path(self) -> str:
        """Get the path for the state persistence file."""
        # Store state file in the same directory as the script
        state_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(state_dir, f"grid_state_{self.symbol}.json")

    def _save_state(self) -> None:
        """Save current strategy state to file for persistence across restarts."""
        try:
            # Convert take_profit_orders to serializable format
            tp_orders_serializable = {}
            for order_id, info in self.take_profit_orders.items():
                tp_orders_serializable[order_id] = {
                    'order_id': info.get('order_id'),
                    'buy_price': info.get('buy_price'),
                    'sell_price': info.get('sell_price'),
                    'quantity': info.get('quantity'),
                    'status': info.get('status'),
                }
            
            state = {
                'reference_price': self.reference_price,
                'take_profit_orders': tp_orders_serializable,
                'total_buys_filled': self.total_buys_filled,
                'total_sells_filled': self.total_sells_filled,
                'total_profit': self.total_profit,
                'filled_lower_prices': list(self.filled_lower_prices),
                'filled_upper_prices': list(self.filled_upper_prices),
                'last_saved': datetime.now().isoformat(),
            }
            
            with open(self._state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"State saved: ref={self.reference_price}, TP orders={len(self.take_profit_orders)}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self) -> bool:
        """
        Load saved state from file.
        
        Returns:
            True if state was loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(self._state_file):
                logger.info("No saved state file found, starting fresh")
                return False
            
            with open(self._state_file, 'r') as f:
                state = json.load(f)
            
            # Restore reference price
            self.reference_price = state.get('reference_price')
            
            # Restore statistics
            self.total_buys_filled = state.get('total_buys_filled', 0)
            self.total_sells_filled = state.get('total_sells_filled', 0)
            self.total_profit = state.get('total_profit', 0.0)
            
            # Restore filled price tracking
            self.filled_lower_prices = set(state.get('filled_lower_prices', []))
            self.filled_upper_prices = set(state.get('filled_upper_prices', []))
            
            # Restore take profit orders
            saved_tp_orders = state.get('take_profit_orders', {})
            for order_id, info in saved_tp_orders.items():
                self.take_profit_orders[order_id] = {
                    'order_id': info.get('order_id'),
                    'buy_price': info.get('buy_price'),
                    'sell_price': info.get('sell_price'),
                    'quantity': info.get('quantity'),
                    'status': info.get('status'),
                }
            
            logger.info(f"State loaded: ref={self.reference_price}, TP orders={len(self.take_profit_orders)}, "
                       f"buys={self.total_buys_filled}, sells={self.total_sells_filled}, profit={self.total_profit}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    # ==================== Sell Only Mode ====================

    def enable_sell_only_mode(self) -> None:
        """
        Enable sell-only mode - cancel all buy orders, stop refilling.
        
        TP sell orders remain active to liquidate existing positions.
        """
        logger.info("Enabling sell-only mode...")
        self.sell_only_mode = True
        
        # Cancel all lower band orders (limit buys)
        cancelled_lower = 0
        for price, order_info in list(self.lower_band_orders.items()):
            order_id = order_info.get('order_id')
            if order_id:
                try:
                    self.client.cancel_spot_order(order_id, self.symbol)
                    cancelled_lower += 1
                except Exception as e:
                    logger.error(f"Failed to cancel lower order {order_id}: {e}")
        self.lower_band_orders.clear()
        
        # Cancel all upper band orders (conditional buys)
        cancelled_upper = 0
        for price, order_info in list(self.upper_band_orders.items()):
            order_id = order_info.get('order_id')
            if order_id:
                try:
                    self.client.cancel_spot_order(order_id, self.symbol)
                    cancelled_upper += 1
                except Exception as e:
                    logger.error(f"Failed to cancel upper order {order_id}: {e}")
        self.upper_band_orders.clear()
        
        logger.info(f"Sell-only mode ENABLED - Cancelled {cancelled_lower} lower + {cancelled_upper} upper buy orders")
        logger.info(f"TP sell orders still active: {len(self.take_profit_orders)}")

    def disable_sell_only_mode(self) -> None:
        """
        Disable sell-only mode and resume normal grid operation.
        
        The next refill cycle will place new buy orders.
        """
        self.sell_only_mode = False
        logger.info("Sell-only mode DISABLED - Resuming normal grid operation")

    def is_sell_only_mode(self) -> bool:
        """Check if sell-only mode is active."""
        return self.sell_only_mode

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
        for i in range(1, self.max_orders_lower + 1):
            price = reference_price - (i * self.grid_spacing)
            price = round_to_tick_size(price, self.tick_size)
            if price > 0:
                lower_prices.append(price)
        
        # Upper band: prices above reference, INCLUDING reference price if i=0
        for i in range(0, self.max_orders_upper):
            price = reference_price + (i * self.grid_spacing)
            price = round_to_tick_size(price, self.tick_size)
            upper_prices.append(price)
        
        return lower_prices, upper_prices

    def initialize_grid(self) -> bool:
        """
        Initialize the grid with orders on both bands.
        
        If saved state exists, restores reference price and re-places TP orders.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing spot buy grid...")
        
        # Get symbol info first (needed for rounding)
        symbol_info = self.client.get_spot_symbol_info(self.symbol)
        if not symbol_info:
            logger.error(f"Failed to get symbol info for {self.symbol}")
            return False
            
        self.tick_size = symbol_info.get('tick_size', 0.01)
        self.qty_step = symbol_info.get('qty_step', 0.0001)
        self.quote_asset = symbol_info.get('quote_asset', 'USDC')
        
        logger.info(f"Symbol Info: tick_size={self.tick_size}, qty_step={self.qty_step}")
        
        # Try to load saved state
        state_loaded = self._load_state()
        
        if state_loaded and self.reference_price:
            logger.info(f"Resuming from saved state with reference price: {self.reference_price}")
            
            # Re-place TP orders that were saved
            # First, get a copy of saved TP orders and clear the dict
            saved_tp_orders = dict(self.take_profit_orders)
            self.take_profit_orders.clear()
            
            # Cancel existing orders and re-place
            self._cancel_all_orders()
            
            # Re-place each TP order
            for old_order_id, order_info in saved_tp_orders.items():
                buy_price = order_info.get('buy_price')
                quantity = order_info.get('quantity')
                if buy_price and quantity:
                    logger.info(f"Re-placing TP order: buy_price={buy_price}, qty={quantity}")
                    self._place_take_profit_order(buy_price, quantity)
            
            logger.info(f"Re-placed {len(self.take_profit_orders)} take profit orders")
            
            # RECOVERY: Check for filled prices without corresponding TP orders
            if self.take_profit_offset > 0:
                # Get all buy prices that have TP orders
                tp_buy_prices = set()
                for order_info in self.take_profit_orders.values():
                    tp_buy_prices.add(round(order_info.get('buy_price', 0), 4))
                
                # Check filled_lower_prices for missing TPs
                for filled_price in list(self.filled_lower_prices):
                    rounded_price = round(filled_price, 4)
                    if rounded_price not in tp_buy_prices:
                        logger.warning(f"RECOVERY: Missing TP for filled lower price {rounded_price}, placing now")
                        if self._place_take_profit_order(rounded_price, self.order_quantity):
                            tp_buy_prices.add(rounded_price)
                
                # Check filled_upper_prices for missing TPs
                for filled_price in list(self.filled_upper_prices):
                    rounded_price = round(filled_price, 4)
                    if rounded_price not in tp_buy_prices:
                        logger.warning(f"RECOVERY: Missing TP for filled upper price {rounded_price}, placing now")
                        if self._place_take_profit_order(rounded_price, self.order_quantity):
                            tp_buy_prices.add(rounded_price)
                
                logger.info(f"Recovery complete. Total TP orders: {len(self.take_profit_orders)}")
        else:
            # No saved state, use current price as reference
            ticker = self.get_ticker()
            if "error" in ticker:
                logger.error(f"Failed to get ticker: {ticker['error']}")
                return False
            
            current_price = float(ticker.get('lastPrice', 0))
            if current_price <= 0:
                logger.error("Invalid current price")
                return False
            
            self.reference_price = current_price
            logger.info(f"Starting fresh with reference price: {current_price}")
            
            # Cancel any existing orders
            self._cancel_all_orders()
        
        # Get current market price for placing orders relative to current price
        ticker = self.get_ticker()
        if "error" in ticker:
            logger.error(f"Failed to get ticker: {ticker['error']}")
            return False
        current_price = float(ticker.get('lastPrice', self.reference_price))
        
        logger.info(f"Reference price: {self.reference_price}, Current price: {current_price}")
        
        # === PLACE LOWER BAND ORDERS ===
        # Find grid levels below current market price, calculated from reference
        lower_prices = []
        i = 1
        while len(lower_prices) < self.max_orders_lower and i < 1000:
            grid_level = round_to_tick_size(
                self.reference_price - (i * self.grid_spacing),
                self.tick_size
            )
            i += 1
            if grid_level <= 0:
                break
            # Only include levels below current price
            if grid_level < current_price:
                # Skip if already filled
                if grid_level not in self.filled_lower_prices:
                    lower_prices.append(grid_level)
        
        logger.info(f"Lower band prices (below current {current_price}): {lower_prices}")
        
        # === PLACE UPPER BAND ORDERS ===
        # Find grid levels above current market price, calculated from reference
        upper_prices = []
        i = 0
        while len(upper_prices) < self.max_orders_upper and i < 1000:
            grid_level = round_to_tick_size(
                self.reference_price + (i * self.grid_spacing),
                self.tick_size
            )
            i += 1
            # Only include levels above current price
            if grid_level > current_price:
                # Skip if already filled
                if grid_level not in self.filled_upper_prices:
                    upper_prices.append(grid_level)
        
        logger.info(f"Upper band prices (above current {current_price}): {upper_prices}")
        
        # Place lower band orders (post-only limit buys)
        placed_lower = 0
        for price in lower_prices:
            if self._place_lower_band_order(price):
                placed_lower += 1
        
        logger.info(f"Placed {placed_lower} lower band orders")
        
        # Place upper band orders (conditional + post-only buys)
        # upper_prices are the LIMIT prices (where we want to buy)
        # The trigger price is calculated as limit + trigger_offset
        placed_upper = 0
        for limit_price in upper_prices:
            if self._place_upper_band_order(limit_price):
                placed_upper += 1
        
        logger.info(f"Placed {placed_upper} upper band orders")
        
        # Save state after initialization
        self._save_state()
        
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

    def _place_upper_band_order(self, limit_price: float) -> bool:
        """
        Place a conditional + post-only buy order in the upper band.
        
        The limit_price is the grid level where we want to buy.
        Trigger activates when price rises to limit_price + trigger_offset.
        
        Args:
            limit_price: The price at which we want to buy (grid level)
            
        Returns:
            True if order placed successfully
        """
        quantity = round_to_tick_size(self.order_quantity, self.qty_step)
        
        # Trigger price is above limit price - order activates when price rises to trigger
        # Then places a limit buy below current price to ensure maker fill
        trigger_price = round_to_tick_size(
            limit_price + self.trigger_offset, 
            self.tick_size
        )
        limit_price = round_to_tick_size(limit_price, self.tick_size)
        
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
        
        logger.info(f"Placing upper band conditional buy: limit={limit_price}, trigger={trigger_price}, qty={quantity}")
        response = self.client.execute_spot_order(order_details)
        
        if "error" in response:
            logger.error(f"Failed to place upper band order: {response['error']}")
            return False
        
        order_id = response.get('orderId')
        if order_id:
            self.upper_band_orders[limit_price] = {
                'order_id': order_id,
                'limit_price': limit_price,
                'trigger_price': trigger_price,
                'quantity': quantity,
                'status': 'open',
                'created_time': datetime.now(),
            }
            self.order_id_to_grid_price[order_id] = limit_price
            self.order_id_to_band[order_id] = 'upper'
            logger.info(f"Upper band order placed: ID={order_id}, limit={limit_price}, trigger={trigger_price}")
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
        # Reduce quantity by trading fee to prevent insufficient balance
        # e.g., if fee is 0.1%, actual received is quantity * (1 - 0.001)
        adjusted_quantity = quantity * (1 - self.trading_fee)
        adjusted_quantity = round_to_tick_size(adjusted_quantity, self.qty_step)
        
        order_details = {
            'symbol': self.symbol,
            'side': 'Sell',
            'orderType': 'Limit',
            'qty': adjusted_quantity,
            'price': tp_price,
            'timeInForce': 'PostOnly',
            'orderFilter': 'Order',
        }
        
        logger.info(f"Placing take profit sell: price={tp_price}, qty={adjusted_quantity} (original: {quantity}, fee: {self.trading_fee*100:.2f}%), buy_price={buy_price}")
        response = self.client.execute_spot_order(order_details)
        
        if "error" in response:
            logger.error(f"Failed to place take profit order: {response['error']}")
            return False
        
        order_id = response.get('orderId')
        if order_id:
            # Store rounded buy_price to match what's in filled sets
            rounded_buy_price = round(buy_price, 4)
            self.take_profit_orders[order_id] = {
                'order_id': order_id,
                'buy_price': rounded_buy_price,
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

    def _handle_buy_fill(self, fill_price: float, quantity: float, band: str) -> None:
        """
        Handle a buy order fill - place take profit and update stats.
        
        Args:
            fill_price: Price at which the buy was filled
            quantity: Quantity that was filled
            band: Which band the order was from ('lower' or 'upper')
        """
        logger.info(f"Buy filled: price={fill_price}, qty={quantity}, band={band}")
    
        # Round price consistently using round() with 4 decimals
        # This must match how we remove in _handle_tp_fill
        rounded_price = round(fill_price, 4)
        
        # Track this price as filled to prevent duplicate refills
        if band == 'lower':
            self.filled_lower_prices.add(rounded_price)
            logger.info(f"Added {rounded_price} to filled_lower_prices (total: {len(self.filled_lower_prices)})")
        elif band == 'upper':
            self.filled_upper_prices.add(rounded_price)
            logger.info(f"Added {rounded_price} to filled_upper_prices (total: {len(self.filled_upper_prices)})")
        
        # Update statistics
        self.total_buys_filled += 1
        
        # Place take profit order if offset is configured
        if self.take_profit_offset > 0:
            self._place_take_profit_order(fill_price, quantity)
        else:
            logger.info("No take profit offset configured, skipping TP order")
        
        # Save state after fill
        self._save_state()

    def _handle_tp_fill(self, order_info: Dict[str, Any]) -> None:
        """
        Handle a take profit order fill - record profit and update stats.
        
        After TP fill, the grid level becomes available again for new orders.
        
        Args:
            order_info: Dictionary with buy_price, sell_price, quantity
        """
        buy_price = order_info.get('buy_price', 0)
        sell_price = order_info.get('sell_price', 0)
        quantity = order_info.get('quantity', 0)
        
        # Calculate profit
        profit = (sell_price - buy_price) * quantity
        
        # Update statistics
        self.total_sells_filled += 1
        self.total_profit += profit
        
        logger.info(f"Take profit filled: bought@{buy_price}, sold@{sell_price}, qty={quantity}, profit={profit:.4f}")
    
        # CYCLIC GRID: Remove buy_price from filled sets so the level can be reused
        # Use round() with 4 decimals to match how filled prices are stored
        rounded_price = round(buy_price, 4)
        if rounded_price in self.filled_lower_prices:
            self.filled_lower_prices.discard(rounded_price)
            logger.info(f"Removed {rounded_price} from filled_lower_prices - level available for reuse")
        if rounded_price in self.filled_upper_prices:
            self.filled_upper_prices.discard(rounded_price)
            logger.info(f"Removed {rounded_price} from filled_upper_prices - level available for reuse")
        
        # Save state after profit recorded
        self._save_state()

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
        for price, order_info in list(self.lower_band_orders.items()):
            order_id = order_info.get('order_id')
            if order_id and order_id not in open_order_ids:
                # Order missing from open list - likely filled
                history = self.client.get_spot_order_history(self.symbol, order_id=order_id)
                
                status = "Unknown"
                filled_qty = 0.0
                
                if isinstance(history, list) and history:
                    order_data = history[0]
                    status = order_data.get('status', 'Unknown')
                    filled_qty = float(order_data.get('filledQty', 0))
                    
                if status in ('Filled', 'PartiallyFilled') or filled_qty > 0:
                    logger.info(f"Lower band order filled: price={price}, ID={order_id}")
                    self._handle_buy_fill(price, order_info['quantity'], 'lower')
                    del self.lower_band_orders[price]
                elif status in ('Cancelled', 'Rejected'):
                    logger.info(f"Lower band order {status}: price={price}")
                    del self.lower_band_orders[price]
                # else: keep in memory silently

        # Check upper band orders for fills using fill history
        # Since conditional orders can't be tracked via order history on Zoomex,
        # we check recent fills for buys at our upper band limit prices
        if self.upper_band_orders:
            fills = self.client.get_spot_fill_history(self.symbol, limit=20)
            if isinstance(fills, list):
                for fill in fills:
                    fill_price = float(fill.get('price', 0))
                    fill_side = fill.get('side', '').upper()
                    fill_id = fill.get('fill_id') or fill.get('id')
                    
                    # Only process buy fills
                    if fill_side != 'BUY':
                        continue
                    
                    # Check if this fill matches any upper band order
                    for limit_price, order_info in list(self.upper_band_orders.items()):
                        # Match if fill price is close to our limit price (within tick size)
                        if abs(fill_price - limit_price) <= self.tick_size * 2:
                            # Check if we haven't already processed this fill
                            if not hasattr(self, '_processed_fill_ids'):
                                self._processed_fill_ids = set()
                            
                            if fill_id and fill_id not in self._processed_fill_ids:
                                self._processed_fill_ids.add(fill_id)
                                logger.info(f"Upper band order filled: limit={limit_price}, fill_price={fill_price}")
                                self._handle_buy_fill(limit_price, order_info['quantity'], 'upper')
                                del self.upper_band_orders[limit_price]
                                break  # Only process one fill per iteration
        
        # Check take profit orders for fills using FILL HISTORY
        # (order history API returns empty for spot orders on Zoomex)
        if self.take_profit_orders:
            fills = self.client.get_spot_fill_history(self.symbol, limit=50)
            if isinstance(fills, list):
                for fill in fills:
                    fill_price = float(fill.get('price', 0))
                    fill_side = fill.get('side', '').upper()
                    fill_id = fill.get('fill_id') or fill.get('id') or fill.get('execId')
                    
                    # Debug: log each fill to see format
                    logger.debug(f"Fill history entry: price={fill_price}, side={fill_side}, id={fill_id}, raw_side={fill.get('side')}")
                    
                    # Only process SELL fills (TP orders are sells)
                    if fill_side not in ('SELL', 'Sell'):
                        continue
                    
                    # Check if this fill matches any TP order's sell price
                    for order_id, order_info in list(self.take_profit_orders.items()):
                        tp_sell_price = float(order_info.get('sell_price', 0))
                        
                        # Match if fill price is close to our TP sell price
                        if abs(fill_price - tp_sell_price) <= self.tick_size * 2:
                            # Check if order is no longer in open orders
                            if order_id not in open_order_ids:
                                # Avoid duplicate processing
                                tp_fill_key = f"tp_{order_id}_{fill_id}"
                                if tp_fill_key not in self._processed_fill_ids:
                                    self._processed_fill_ids.add(tp_fill_key)
                                    logger.info(f"TP sell filled via fill history: ID={order_id}, sell_price={fill_price}, buy_price={order_info.get('buy_price')}")
                                    self._handle_tp_fill(order_info)
                                    del self.take_profit_orders[order_id]
                                    break

    def _refill_grid_orders(self) -> None:
        """
        Refill missing grid orders using reference-based grid levels.
        
        Grid levels are always calculated from reference_price:
        - Lower band: ref - (i * spacing) for levels below current price
        - Upper band: ref + (i * spacing) for levels above current price
        
        This ensures orders are placed near current market price even after
        significant price moves, while maintaining consistent grid spacing.
        """
        if not self.reference_price:
            return
        
        # Don't refill if in sell-only mode
        if self.sell_only_mode:
            return
        
        # Get current market price for determining which levels to use
        ticker = self.get_ticker()
        if "error" in ticker:
            return
        current_price = float(ticker.get('lastPrice', self.reference_price))
        
        # Count current orders
        lower_count = len(self.lower_band_orders)
        upper_count = len(self.upper_band_orders)
        
        # === REBALANCE LOWER BAND ===
        # If at max orders but a closer level is available, cancel furthest and refill closer
        if lower_count >= self.max_orders_lower and self.lower_band_orders:
            # Find the closest available grid level below current price
            closest_available = None
            i = 1
            max_check = 100
            while i < max_check:
                grid_level = round(self.reference_price - (i * self.grid_spacing), 4)
                i += 1
                if grid_level <= 0 or grid_level >= current_price:
                    continue
                # Check if this level is available (not filled, not already ordered)
                if grid_level not in self.filled_lower_prices:
                    order_exists = any(abs(p - grid_level) < 0.00001 for p in self.lower_band_orders.keys())
                    if not order_exists:
                        closest_available = grid_level
                        break
            
            if closest_available:
                # Find the furthest existing order
                furthest_price = min(self.lower_band_orders.keys())
                
                # Only rebalance if closest available is closer than furthest existing
                if closest_available > furthest_price:
                    logger.info(f"Rebalancing lower band: cancel {furthest_price}, will place {closest_available}")
                    order_info = self.lower_band_orders[furthest_price]
                    order_id = order_info.get('order_id')
                    if order_id:
                        try:
                            self.client.cancel_spot_order(order_id, self.symbol)
                            del self.lower_band_orders[furthest_price]
                            lower_count -= 1
                            logger.info(f"Cancelled furthest lower order at {furthest_price}")
                        except Exception as e:
                            logger.error(f"Failed to cancel for rebalance: {e}")
        
        # === REBALANCE UPPER BAND ===
        # If at max orders but a closer level is available, cancel furthest and refill closer
        if upper_count >= self.max_orders_upper and self.upper_band_orders:
            # Find the closest available grid level above current price
            closest_available = None
            i = 0
            max_check = 100
            while i < max_check:
                grid_level = round(self.reference_price + (i * self.grid_spacing), 4)
                i += 1
                if grid_level <= current_price:
                    continue
                # Check if this level is available (not filled, not already ordered)
                if grid_level not in self.filled_upper_prices:
                    order_exists = any(abs(p - grid_level) < 0.00001 for p in self.upper_band_orders.keys())
                    if not order_exists:
                        closest_available = grid_level
                        break
            
            if closest_available:
                # Find the furthest existing order (highest price for upper band)
                furthest_price = max(self.upper_band_orders.keys())
                
                # Only rebalance if closest available is closer than furthest existing
                if closest_available < furthest_price:
                    logger.info(f"Rebalancing upper band: cancel {furthest_price}, will place {closest_available}")
                    order_info = self.upper_band_orders[furthest_price]
                    order_id = order_info.get('order_id')
                    if order_id:
                        try:
                            self.client.cancel_spot_order(order_id, self.symbol)
                            del self.upper_band_orders[furthest_price]
                            upper_count -= 1
                            logger.info(f"Cancelled furthest upper order at {furthest_price}")
                        except Exception as e:
                            logger.error(f"Failed to cancel for rebalance: {e}")
        
        # === REFILL LOWER BAND ===
        # Find grid levels below current price, calculated from reference
        if lower_count < self.max_orders_lower:
            orders_needed = self.max_orders_lower - lower_count
            orders_placed = 0
            
            # Calculate grid levels below current price from reference
            # Start from i=1 (first level below reference)
            i = 1
            max_iterations = 1000  # Safety limit to prevent infinite loop
            
            while orders_placed < orders_needed and i < max_iterations:
                # Calculate grid level from reference
                grid_level = round(self.reference_price - (i * self.grid_spacing), 4)
                i += 1
                
                if grid_level <= 0:
                    break
                
                # Skip levels at or above current price
                if grid_level >= current_price:
                    continue
                
                # Skip if already filled
                if grid_level in self.filled_lower_prices:
                    logger.debug(f"Skipping filled lower level {grid_level}")
                    continue
                
                # Skip if order already exists at this price
                rounded_level = round(grid_level, 4)
                order_exists = any(abs(p - grid_level) < 0.00001 for p in self.lower_band_orders.keys())
                if order_exists:
                    continue
                
                # Skip if order exists on exchange at this price
                if any(abs(p - grid_level) < 0.00001 for p in self.active_order_prices):
                    continue
                
                # Place the order
                logger.info(f"Placing lower band at grid level {grid_level} (current price: {current_price})")
                if self._place_lower_band_order(grid_level):
                    orders_placed += 1
                    lower_count += 1
        
        # === REFILL UPPER BAND ===
        # Find grid levels above current price, calculated from reference
        if upper_count < self.max_orders_upper:
            orders_needed = self.max_orders_upper - upper_count
            orders_placed = 0
            
            # Calculate grid levels above current price from reference
            # i=0 means at reference, i=1 means reference + spacing, etc.
            i = 0
            max_iterations = 1000
            
            while orders_placed < orders_needed and i < max_iterations:
                # Calculate grid level from reference
                grid_level = round(self.reference_price + (i * self.grid_spacing), 4)
                i += 1
                
                # Skip levels at or below current price
                if grid_level <= current_price:
                    continue
                
                # Skip if already filled
                if grid_level in self.filled_upper_prices:
                    logger.debug(f"Skipping filled upper level {grid_level}")
                    continue
                
                # Skip if order already exists at this limit
                order_exists = any(abs(p - grid_level) < 0.00001 for p in self.upper_band_orders.keys())
                if order_exists:
                    continue
                
                # Skip if order exists on exchange at this price
                if any(abs(p - grid_level) < 0.00001 for p in self.active_order_prices):
                    continue
                
                # Place the order (trigger = limit + offset)
                logger.info(f"Placing upper band at grid level {grid_level} (current price: {current_price})")
                if self._place_upper_band_order(grid_level):
                    orders_placed += 1
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
            # Save state before canceling orders so we can restore on restart
            self._save_state()
            logger.info(f"State saved to {self._state_file}")
            self._cancel_all_orders()
            logger.info(self._get_extra_summary_sections())
            logger.info("Spot Buy Grid Strategy stopped")
