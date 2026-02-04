"""
Trading Signals Module - Technical Analysis & Trading Recommendations
Uses multiple indicators: SMA, RSI, MACD, Bollinger Bands, Volume
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class TradingSignalAnalyzer:
    """Generate trading signals based on technical indicators"""
    def __init__(self, price_history):
        """
        Initialize with historical price data
        Args:
            price_history: DataFrame with 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = price_history.copy().sort_index()
        self.signals = {}
        self.indicators = {}
    
    def calculate_sma(self, period):
        """Calculate Simple Moving Average"""
        return self.df['close'].rolling(window=period).mean()
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index (0-100)"""
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.df['close'].ewm(span=fast).mean()
        ema_slow = self.df['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(period)
        std = self.df['close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range (volatility indicator)"""
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_volume_trend(self, period=20):
        """Calculate volume trend (current volume vs average)"""
        avg_volume = self.df['volume'].rolling(window=period).mean()
        volume_ratio = self.df['volume'] / avg_volume
        return volume_ratio
    
    def sma_crossover_signal(self):
        """SMA Crossover Strategy (20, 50, 200)"""
        sma20 = self.calculate_sma(20)
        sma50 = self.calculate_sma(50)
        sma200 = self.calculate_sma(200)
        
        current_price = self.df['close'].iloc[-1]
        prev_price = self.df['close'].iloc[-2] if len(self.df) > 1 else current_price
        
        current_sma20 = sma20.iloc[-1]
        current_sma50 = sma50.iloc[-1]
        current_sma200 = sma200.iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        reason = []
        
        # Golden Cross (SMA20 > SMA50 > SMA200) = BULLISH
        if current_sma20 > current_sma50 > current_sma200:
            signal = "BUY"
            confidence = 0.85
            reason.append("Golden Cross: SMA20 > SMA50 > SMA200")
        
        # Death Cross (SMA20 < SMA50 < SMA200) = BEARISH
        elif current_sma20 < current_sma50 < current_sma200:
            signal = "SELL"
            confidence = 0.85
            reason.append("Death Cross: SMA20 < SMA50 < SMA200")
        
        # SMA20 breaking above SMA50
        elif current_sma20 > current_sma50 and sma20.iloc[-2] <= sma50.iloc[-2]:
            signal = "BUY"
            confidence = 0.75
            reason.append("SMA20 crossed above SMA50")
        
        # SMA20 breaking below SMA50
        elif current_sma20 < current_sma50 and sma20.iloc[-2] >= sma50.iloc[-2]:
            signal = "SELL"
            confidence = 0.75
            reason.append("SMA20 crossed below SMA50")
        
        # Price above all SMAs = BULLISH
        elif current_price > current_sma20 > current_sma50 > current_sma200:
            signal = "BUY"
            confidence = 0.70
            reason.append("Price above all SMAs (bullish trend)")
        
        # Price below all SMAs = BEARISH
        elif current_price < current_sma20 < current_sma50 < current_sma200:
            signal = "SELL"
            confidence = 0.70
            reason.append("Price below all SMAs (bearish trend)")
        
        else:
            reason.append("Mixed SMA signals - consolidation phase")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'sma20': round(current_sma20, 2),
            'sma50': round(current_sma50, 2),
            'sma200': round(current_sma200, 2),
            'price': round(current_price, 2)
        }
    
    def rsi_signal(self, period=14):
        """RSI (Relative Strength Index) Signal"""
        rsi = self.calculate_rsi(period)
        current_rsi = rsi.iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        reason = []
        
        if current_rsi > 70:
            signal = "SELL"
            confidence = 0.65
            reason.append(f"RSI ({current_rsi:.2f}) > 70 - Overbought")
        elif current_rsi < 30:
            signal = "BUY"
            confidence = 0.65
            reason.append(f"RSI ({current_rsi:.2f}) < 30 - Oversold")
        elif current_rsi > 50:
            reason.append(f"RSI ({current_rsi:.2f}) - Slightly bullish")
        else:
            reason.append(f"RSI ({current_rsi:.2f}) - Slightly bearish")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'rsi': round(current_rsi, 2)
        }
    
    def macd_signal(self):
        """MACD (Moving Average Convergence Divergence) Signal"""
        macd_line, signal_line, histogram = self.calculate_macd()
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        signal = "HOLD"
        confidence = 0.5
        reason = []
        
        # MACD above signal line and positive histogram = BULLISH
        if current_macd > current_signal and current_histogram > 0:
            signal = "BUY"
            confidence = 0.70
            reason.append("MACD above signal line (bullish)")
        
        # MACD below signal line and negative histogram = BEARISH
        elif current_macd < current_signal and current_histogram < 0:
            signal = "SELL"
            confidence = 0.70
            reason.append("MACD below signal line (bearish)")
        
        # MACD crossing above signal line = BUY
        elif current_macd > current_signal and prev_histogram <= 0:
            signal = "BUY"
            confidence = 0.75
            reason.append("MACD crossed above signal line")
        
        # MACD crossing below signal line = SELL
        elif current_macd < current_signal and prev_histogram >= 0:
            signal = "SELL"
            confidence = 0.75
            reason.append("MACD crossed below signal line")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'macd': round(current_macd, 4),
            'signal_line': round(current_signal, 4),
            'histogram': round(current_histogram, 4)
        }
    
    def bollinger_bands_signal(self, period=20):
        """Bollinger Bands Signal"""
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(period)
        
        current_price = self.df['close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_middle = middle_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        reason = []
        
        # Price touching upper band = OVERBOUGHT
        if current_price >= current_upper * 0.98:
            signal = "SELL"
            confidence = 0.65
            reason.append("Price near upper Bollinger Band (overbought)")
        
        # Price touching lower band = OVERSOLD
        elif current_price <= current_lower * 1.02:
            signal = "BUY"
            confidence = 0.65
            reason.append("Price near lower Bollinger Band (oversold)")
        
        # Price above middle band = BULLISH
        elif current_price > current_middle:
            reason.append("Price above middle Bollinger Band (bullish)")
        else:
            reason.append("Price below middle Bollinger Band (bearish)")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'upper_band': round(current_upper, 2),
            'middle_band': round(current_middle, 2),
            'lower_band': round(current_lower, 2),
            'price': round(current_price, 2)
        }
    
    def volume_signal(self, period=20):
        """Volume Analysis Signal"""
        volume_ratio = self.calculate_volume_trend(period)
        current_volume_ratio = volume_ratio.iloc[-1]
        
        signal = "HOLD"
        confidence = 0.5
        reason = []
        
        if current_volume_ratio > 1.5:
            reason.append(f"High volume ({current_volume_ratio:.2f}x average) - Strong signal")
            confidence = 0.7
        elif current_volume_ratio > 1.2:
            reason.append(f"Above average volume ({current_volume_ratio:.2f}x) - Good signal")
            confidence = 0.6
        elif current_volume_ratio < 0.7:
            reason.append(f"Low volume ({current_volume_ratio:.2f}x) - Weak signal")
            confidence = 0.4
        else:
            reason.append(f"Normal volume ({current_volume_ratio:.2f}x)")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'volume_ratio': round(current_volume_ratio, 2)
        }
    
    def generate_comprehensive_signal(self):
        """Combine all signals and generate final recommendation"""
        # Calculate all signals
        sma_signal = self.sma_crossover_signal()
        rsi_signal = self.rsi_signal()
        macd_signal = self.macd_signal()
        bb_signal = self.bollinger_bands_signal()
        vol_signal = self.volume_signal()
        
        # Aggregate signals
        signals_list = []
        confidences = []
        
        # Collect non-HOLD signals with weights
        for sig_name, sig_data in [('SMA', sma_signal), ('RSI', rsi_signal), 
                                    ('MACD', macd_signal), ('BB', bb_signal)]:
            if sig_data['signal'] != 'HOLD':
                signals_list.append((sig_data['signal'], sig_data['confidence']))
                confidences.append(sig_data['confidence'])
        
        # Determine final signal
        final_signal = "HOLD"
        final_confidence = 0.5
        
        if signals_list:
            buy_score = sum(c for s, c in signals_list if s == 'BUY') / len(signals_list)
            sell_score = sum(c for s, c in signals_list if s == 'SELL') / len(signals_list)
            
            if buy_score > sell_score and buy_score > 0.6:
                final_signal = "BUY"
                final_confidence = min(float(buy_score), 0.95)
            elif sell_score > buy_score and sell_score > 0.6:
                final_signal = "SELL"
                final_confidence = min(float(sell_score), 0.95)
        else:
            final_signal = "HOLD"
            final_confidence = 0.5
        
        return {
            'final_signal': final_signal,
            'final_confidence': round(final_confidence, 2),
            'sma': sma_signal,
            'rsi': rsi_signal,
            'macd': macd_signal,
            'bollinger_bands': bb_signal,
            'volume': vol_signal,
            'signals_count': len(signals_list),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class StrategySimulator:
    """Simple strategy state machine simulator with risk management.

    States: FLAT -> ENTER -> HOLD -> EXIT -> FLAT
    Uses ATR for volatility-based stop-loss, fixed risk-per-trade position sizing,
    and optional take-profit multiplier.
    """

    def __init__(self, price_df, initial_capital=1_000_000, risk_per_trade=0.01,
                 atr_period=14, atr_sl_multiplier=3.0, tp_atr_multiplier=2.0,
                 min_confidence=0.6, max_positions=1):
        self.price_df = price_df.copy().sort_index()
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.equity_curve = []
        self.positions = []
        self.trades = []
        self.risk_per_trade = float(risk_per_trade)
        self.atr_period = int(atr_period)
        self.atr_sl_multiplier = float(atr_sl_multiplier)
        self.tp_atr_multiplier = float(tp_atr_multiplier)
        self.min_confidence = float(min_confidence)
        self.max_positions = int(max_positions)

    def _calc_atr(self, df, period=None):
        p = period or self.atr_period
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=p).mean()
        return atr

    def simulate(self):
        state = 'FLAT'
        current_position = None
        analyzer = None

        # Prepare ATR series once
        atr_series = self._calc_atr(self.price_df)

        portfolio_value = self.cash
        self.equity_curve = []

        dates = list(self.price_df.index)

        for i, date in enumerate(dates):
            # require enough history
            if i < 30:
                # maintain equity curve
                self.equity_curve.append({'date': date, 'equity': portfolio_value})
                continue

            # create a temporary analyzer up to current index to compute signals
            slice_df = self.price_df.iloc[: i + 1].copy()
            temp_an = TradingSignalAnalyzer(slice_df)
            sigs = temp_an.generate_comprehensive_signal()
            price = float(slice_df['close'].iloc[-1])
            atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

            # State machine
            if state == 'FLAT':
                # Enter if BUY signal and confidence threshold met
                if sigs['final_signal'] == 'BUY' and sigs['final_confidence'] >= self.min_confidence and len(self.positions) < self.max_positions:
                    # Position sizing: risk per trade -> quantity
                    risk_amount = self.initial_capital * self.risk_per_trade
                    stop_loss = price - max(atr * self.atr_sl_multiplier, atr * 0.5 if atr > 0 else price * 0.01)
                    if stop_loss >= price:
                        # invalid stop, skip
                        self.equity_curve.append({'date': date, 'equity': portfolio_value})
                        continue
                    per_share_risk = price - stop_loss
                    qty = math.floor(risk_amount / per_share_risk) if per_share_risk > 0 else 0
                    if qty <= 0:
                        self.equity_curve.append({'date': date, 'equity': portfolio_value})
                        continue

                    # Cap by available cash
                    cost = qty * price
                    if cost > self.cash:
                        qty = math.floor(self.cash / price)
                        cost = qty * price

                    if qty <= 0:
                        self.equity_curve.append({'date': date, 'equity': portfolio_value})
                        continue

                    take_profit = price + (atr * self.tp_atr_multiplier if atr > 0 else price * 0.02)

                    current_position = {
                        'entry_date': date,
                        'entry_price': price,
                        'qty': int(qty),
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'state': 'HOLD'
                    }

                    # Debit cash
                    self.cash -= cost
                    self.positions.append(current_position)
                    state = 'HOLD'

            elif state == 'HOLD' and current_position is not None:
                # Check stop loss or take profit
                if price <= current_position['stop_loss']:
                    # Stop loss hit
                    exit_price = price
                    qty = current_position['qty']
                    pnl = (exit_price - current_position['entry_price']) * qty
                    self.cash += exit_price * qty
                    self.trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'qty': qty,
                        'pnl': pnl
                    })
                    self.positions.remove(current_position)
                    current_position = None
                    state = 'FLAT'

                elif price >= current_position['take_profit']:
                    # Take profit
                    exit_price = price
                    qty = current_position['qty']
                    pnl = (exit_price - current_position['entry_price']) * qty
                    self.cash += exit_price * qty
                    self.trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'qty': qty,
                        'pnl': pnl
                    })
                    self.positions.remove(current_position)
                    current_position = None
                    state = 'FLAT'

                else:
                    # optional: exit on strong SELL signal
                    if sigs['final_signal'] == 'SELL' and sigs['final_confidence'] >= self.min_confidence:
                        exit_price = price
                        qty = current_position['qty']
                        pnl = (exit_price - current_position['entry_price']) * qty
                        self.cash += exit_price * qty
                        self.trades.append({
                            'entry_date': current_position['entry_date'],
                            'exit_date': date,
                            'entry_price': current_position['entry_price'],
                            'exit_price': exit_price,
                            'qty': qty,
                            'pnl': pnl
                        })
                        self.positions.remove(current_position)
                        current_position = None
                        state = 'FLAT'

            # Update equity
            position_value = sum([p['qty'] * price for p in self.positions])
            portfolio_value = self.cash + position_value
            self.equity_curve.append({'date': date, 'equity': portfolio_value})

        # Finish: if still holding, close at last price
        if current_position is not None:
            last_price = float(self.price_df['close'].iloc[-1])
            qty = current_position['qty']
            exit_price = last_price
            pnl = (exit_price - current_position['entry_price']) * qty
            self.cash += exit_price * qty
            self.trades.append({
                'entry_date': current_position['entry_date'],
                'exit_date': self.price_df.index[-1],
                'entry_price': current_position['entry_price'],
                'exit_price': exit_price,
                'qty': qty,
                'pnl': pnl
            })
            self.positions.remove(current_position)

        results = self._calc_performance()
        results['trades'] = self.trades
        results['equity_curve'] = pd.DataFrame(self.equity_curve).set_index('date')
        return results

    def _calc_performance(self):
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        equity_df = equity_df.sort_index()
        if equity_df.empty:
            return {'total_return': 0.0, 'cagr': 0.0, 'num_trades': 0, 'win_rate': 0.0, 'max_drawdown': 0.0}

        start = equity_df['equity'].iloc[0]
        end = equity_df['equity'].iloc[-1]
        total_return = (end - start) / start if start != 0 else 0.0

        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = max(days / 365.25, 1/365.25)
        cagr = (1 + total_return) ** (1/years) - 1

        pnl_list = [t['pnl'] for t in self.trades]
        wins = [p for p in pnl_list if p > 0]
        win_rate = (len(wins) / len(pnl_list)) if pnl_list else 0.0

        # Calculate max drawdown
        eq = equity_df['equity']
        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / roll_max
        max_dd = drawdown.min() if not drawdown.empty else 0.0

        # Simple Sharpe ratio (daily returns)
        daily_ret = eq.pct_change().dropna()
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * (252 ** 0.5) if not daily_ret.empty else 0.0

        return {
            'initial_capital': self.initial_capital,
            'ending_capital': float(self.cash),
            'total_return': float(total_return),
            'cagr': float(cagr),
            'num_trades': len(pnl_list),
            'win_rate': float(win_rate),
            'avg_win': float(np.mean([p for p in pnl_list if p > 0]) ) if any(p > 0 for p in pnl_list) else 0.0,
            'avg_loss': float(np.mean([p for p in pnl_list if p <= 0]) ) if any(p <= 0 for p in pnl_list) else 0.0,
            'max_drawdown': float(max_dd),
            'sharpe': float(sharpe)
        }


def evaluate_universe(price_dict, **sim_kwargs):
    """Evaluate a universe of symbols.

    price_dict: dict symbol -> DataFrame(price history)
    sim_kwargs: passed to StrategySimulator
    Returns: summary DataFrame and detail per-symbol results
    """
    summaries = []
    details = {}
    for sym, df in price_dict.items():
        try:
            sim = StrategySimulator(df, **sim_kwargs)
            res = sim.simulate()
            perf = {
                'symbol': sym,
                'total_return': res.get('total_return', 0.0),
                'cagr': res.get('cagr', 0.0),
                'num_trades': res.get('num_trades', 0),
                'win_rate': res.get('win_rate', 0.0),
                'max_drawdown': res.get('max_drawdown', 0.0),
                'sharpe': res.get('sharpe', 0.0)
            }
            summaries.append(perf)
            details[sym] = res
        except Exception:
            summaries.append({'symbol': sym, 'total_return': None, 'cagr': None, 'num_trades': 0, 'win_rate': None, 'max_drawdown': None, 'sharpe': None})

    summary_df = pd.DataFrame(summaries).set_index('symbol')
    return summary_df, details
