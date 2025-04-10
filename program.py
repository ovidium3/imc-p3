from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import statistics

class Trader:
    def __init__(self):
        # Position limits for the three products
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Parameters for strategies
        self.squid_window = 15  # Window for calculating mean and std dev
        self.squid_threshold = 1.75  # Z-score threshold for mean reversion
        self.kelp_short_window = 5
        self.kelp_long_window = 15
        
        # Stable value for Rainforest Resin (based on competition description)
        self.resin_fair_value = 10000

    def run(self, state: TradingState):
        # Initialize result and conversions
        result = {}
        conversions = 0
        
        # Initialize trader data structure if it doesn't exist
        trader_data = {"prices": {}}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except json.JSONDecodeError:
                # If data is corrupted, start fresh
                trader_data = {"prices": {}}
        
        # Process each product in the order depths
        for product in state.order_depths:
            # Skip products we don't have strategies for
            if product not in self.position_limits:
                continue
                
            # Get the order depth for this product
            order_depth = state.order_depths[product]
            
            # Initialize orders list for this product
            orders: List[Order] = []
            
            # Get current position
            current_position = state.position.get(product, 0)
            
            # Calculate mid price if available
            mid_price = self.calculate_mid_price(order_depth)
            
            # Update price history
            if mid_price is not None:
                if product not in trader_data["prices"]:
                    trader_data["prices"][product] = []
                
                trader_data["prices"][product].append({
                    "timestamp": state.timestamp,
                    "price": mid_price
                })
                
                # Keep only the most recent data points to limit memory usage
                max_history = max(50, self.squid_window, self.kelp_long_window)
                if len(trader_data["prices"][product]) > max_history:
                    trader_data["prices"][product] = trader_data["prices"][product][-max_history:]
            
            # Skip trading if we don't have market data to make decisions
            if not order_depth.buy_orders and not order_depth.sell_orders:
                continue
                
            # Calculate fair price based on product strategy
            if product == "RAINFOREST_RESIN":
                fair_price = self.calculate_resin_price(order_depth)
            elif product == "KELP":
                fair_price = self.calculate_kelp_price(order_depth, trader_data)
            elif product == "SQUID_INK":
                fair_price = self.calculate_squid_price(order_depth, trader_data)
            else:
                continue  # Skip unknown products
                
            # Skip if we couldn't determine a fair price
            if fair_price is None:
                continue
            
            # Execute trading strategy
            orders = self.generate_orders(product, order_depth, fair_price, current_position)
            
            # Add orders to result
            if orders:
                result[product] = orders
        
        # Return the result
        return result, conversions, json.dumps(trader_data)

    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from the order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        return (best_bid + best_ask) / 2

    def calculate_resin_price(self, order_depth: OrderDepth) -> float:
        """Calculate the fair price for RAINFOREST_RESIN.
        Strategy: Basic market making around the stable value."""
        return self.resin_fair_value

    def calculate_kelp_price(self, order_depth: OrderDepth, trader_data: dict) -> float:
        """Calculate the fair price for KELP.
        Strategy: Trend following using moving averages."""
        price_history = trader_data["prices"].get("KELP", [])
        
        # Not enough data, use mid price
        if len(price_history) < self.kelp_short_window:
            return self.calculate_mid_price(order_depth)
        
        # Calculate short-term and long-term moving averages
        short_window = [entry["price"] for entry in price_history[-self.kelp_short_window:]]
        short_ma = sum(short_window) / len(short_window)
        
        # If we have enough data for the long MA
        if len(price_history) >= self.kelp_long_window:
            long_window = [entry["price"] for entry in price_history[-self.kelp_long_window:]]
            long_ma = sum(long_window) / len(long_window)
            
            # Trend-following: in uptrend, set fair price slightly higher; in downtrend, slightly lower
            if short_ma > long_ma:
                return short_ma * 1.01  # Uptrend
            else:
                return short_ma * 0.99  # Downtrend
        
        # Not enough data for long MA, just use short MA
        return short_ma

    def calculate_squid_price(self, order_depth: OrderDepth, trader_data: dict) -> float:
        """Calculate the fair price for SQUID_INK.
        Strategy: Mean reversion based on volatility."""
        price_history = trader_data["prices"].get("SQUID_INK", [])
        
        # Not enough data, use mid price
        if len(price_history) < self.squid_window:
            return self.calculate_mid_price(order_depth)
        
        # Get recent prices
        recent_prices = [entry["price"] for entry in price_history[-self.squid_window:]]
        
        # Calculate mean and standard deviation
        mean_price = statistics.mean(recent_prices)
        
        # Calculate standard deviation with a safety check
        try:
            std_dev = statistics.stdev(recent_prices)
        except statistics.StatisticsError:
            # Not enough variety in prices
            return mean_price
        
        # Get current price
        current_price = recent_prices[-1]
        
        # Calculate z-score (number of standard deviations from mean)
        if std_dev > 0:
            z_score = (current_price - mean_price) / std_dev
        else:
            return mean_price
        
        # Mean reversion strategy: if price is too high, expect it to fall; if too low, expect it to rise
        if z_score > self.squid_threshold:
            # Price is significantly above mean, expect it to fall
            return mean_price - (0.5 * std_dev)
        elif z_score < -self.squid_threshold:
            # Price is significantly below mean, expect it to rise
            return mean_price + (0.5 * std_dev)
        else:
            # Price is within normal range, expect it to move toward mean
            return mean_price

    def generate_orders(self, product: str, order_depth: OrderDepth, fair_price: float, current_position: int) -> List[Order]:
        """Generate orders based on the fair price and current position."""
        orders = []
        remaining_buy_capacity = self.position_limits[product] - current_position
        remaining_sell_capacity = self.position_limits[product] + current_position
        
        # Process sell orders (we buy from them)
        sell_orders_sorted = sorted(order_depth.sell_orders.items())  # Sort by price ascending
        for price, volume in sell_orders_sorted:
            # Only buy if the price is below our fair price
            if price < fair_price and remaining_buy_capacity > 0:
                buy_volume = min(-volume, remaining_buy_capacity)
                if buy_volume > 0:
                    orders.append(Order(product, price, buy_volume))
                    remaining_buy_capacity -= buy_volume
        
        # Process buy orders (we sell to them)
        buy_orders_sorted = sorted(order_depth.buy_orders.items(), reverse=True)  # Sort by price descending
        for price, volume in buy_orders_sorted:
            # Only sell if the price is above our fair price
            if price > fair_price and remaining_sell_capacity > 0:
                sell_volume = min(volume, remaining_sell_capacity)
                if sell_volume > 0:
                    orders.append(Order(product, price, -sell_volume))
                    remaining_sell_capacity -= sell_volume
        
        return orders
