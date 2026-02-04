from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
import time
import hashlib
from datetime import datetime, timedelta
from demo_sp import BusinessAnalyzer
from trading_signals import TradingSignalAnalyzer, evaluate_universe, StrategySimulator
from functools import wraps
import warnings
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- CONFIGURATION (The "Brain" Setup) ---
app.secret_key = 'your-secret-key-here' # Needed for session/login

# Database Config (Matches your docker-compose service name 'db')
app.config['MYSQL_HOST'] = 'db'
app.config['MYSQL_USER'] = 'user'
app.config['MYSQL_PASSWORD'] = 'userpassword'
app.config['MYSQL_DB'] = 'vnstock_db'
# Connection pooling and timeout settings
app.config['MYSQL_CONNECT_TIMEOUT'] = 10
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Add built-in functions to Jinja2 context
app.jinja_env.globals.update(min=min)
app.jinja_env.globals.update(max=max)

# Database retry decorator
def db_retry(max_attempts=3, delay=1):
    """Retry database operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = delay * (2 ** attempt)
                    print(f"Database error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator


# --- PASSWORD HASHING FUNCTIONS ---
def hash_password(password):
    """Hash password using MD5"""
    return hashlib.md5(password.encode('utf-8')).hexdigest()

def verify_password(password, hashed_password):
    """Verify password against MD5 hash"""
    return hashlib.md5(password.encode('utf-8')).hexdigest() == hashed_password


# --- TRADING SIGNAL STORAGE FUNCTIONS ---
@db_retry(max_attempts=3, delay=1)
def save_trading_signal(user_id, symbol, company_name, trading_signal_data):
    """Save trading signal to database"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        signal_date = datetime.now().date()
        
        # Extract signal data
        final_signal = trading_signal_data['final_signal']
        final_confidence = trading_signal_data['final_confidence']
        
        sma = trading_signal_data.get('sma', {})
        rsi = trading_signal_data.get('rsi', {})
        macd = trading_signal_data.get('macd', {})
        bb = trading_signal_data.get('bollinger_bands', {})
        vol = trading_signal_data.get('volume', {})
        
        cursor.execute('''
            INSERT INTO trading_signals 
            (user_id, symbol, company_name, signal_date, final_signal, final_confidence,
             sma_signal, sma_confidence, sma_price, sma20, sma50, sma200,
             rsi_signal, rsi_confidence, rsi_value,
             macd_signal, macd_confidence, macd_value, signal_line, histogram,
             bb_signal, bb_confidence, bb_upper, bb_middle, bb_lower,
             volume_ratio)
            VALUES (%s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s)
            ON DUPLICATE KEY UPDATE
            final_signal = VALUES(final_signal),
            final_confidence = VALUES(final_confidence),
            sma_signal = VALUES(sma_signal),
            sma_confidence = VALUES(sma_confidence),
            rsi_signal = VALUES(rsi_signal),
            rsi_confidence = VALUES(rsi_confidence),
            macd_signal = VALUES(macd_signal),
            macd_confidence = VALUES(macd_confidence),
            bb_signal = VALUES(bb_signal),
            bb_confidence = VALUES(bb_confidence),
            volume_ratio = VALUES(volume_ratio)
        ''', (
            user_id, symbol, company_name, signal_date, final_signal, final_confidence,
            sma.get('signal'), sma.get('confidence'), sma.get('price'), 
            sma.get('sma20'), sma.get('sma50'), sma.get('sma200'),
            rsi.get('signal'), rsi.get('confidence'), rsi.get('rsi'),
            macd.get('signal'), macd.get('confidence'), macd.get('macd'),
            macd.get('signal_line'), macd.get('histogram'),
            bb.get('signal'), bb.get('confidence'), bb.get('upper_band'),
            bb.get('middle_band'), bb.get('lower_band'),
            vol.get('volume_ratio')
        ))
        
        mysql.connection.commit()
        return True
    except Exception as e:
        print(f"Error saving trading signal: {e}")
        return False

@db_retry(max_attempts=3, delay=1)
def get_trading_signals_history(user_id, symbol, limit=30):
    """Get trading signal history for a stock"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''
            SELECT * FROM trading_signals 
            WHERE user_id = %s AND symbol = %s
            ORDER BY signal_date DESC
            LIMIT %s
        ''', (user_id, symbol, limit))
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting trading signal history: {e}")
        return []

@db_retry(max_attempts=3, delay=1)
def save_recommendation(user_id, symbol, company_name, action, confidence, reason, 
                       entry_price=None, target_price=None, stop_loss=None):
    """Save trading recommendation"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''
            INSERT INTO trading_recommendations 
            (user_id, symbol, company_name, action, confidence_score, reason,
             entry_price, target_price, stop_loss, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (user_id, symbol, company_name, action, confidence, reason,
              entry_price, target_price, stop_loss, 'ACTIVE'))
        
        mysql.connection.commit()
        return cursor.lastrowid
    except Exception as e:
        print(f"Error saving recommendation: {e}")
        return None

@db_retry(max_attempts=3, delay=1)
def get_user_recommendations(user_id, status='ACTIVE', limit=20):
    """Get user's active recommendations"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''
            SELECT * FROM trading_recommendations 
            WHERE user_id = %s AND status = %s
            ORDER BY created_at DESC
            LIMIT %s
        ''', (user_id, status, limit))
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

@db_retry(max_attempts=3, delay=1)
def update_recommendation_status(recommendation_id, status, realized_price=None):
    """Update recommendation status and calculate P&L"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        if realized_price:
            # Get the recommendation first
            cursor.execute(
                'SELECT entry_price FROM trading_recommendations WHERE id = %s',
                (recommendation_id,)
            )
            rec = cursor.fetchone()
            if rec and rec['entry_price']:
                pnl = realized_price - rec['entry_price']
                pnl_pct = (pnl / rec['entry_price']) * 100 if rec['entry_price'] != 0 else 0
                
                cursor.execute('''
                    UPDATE trading_recommendations 
                    SET status = %s, realized_price = %s, pnl = %s, pnl_percentage = %s
                    WHERE id = %s
                ''', (status, realized_price, pnl, pnl_pct, recommendation_id))
            else:
                cursor.execute(
                    'UPDATE trading_recommendations SET status = %s WHERE id = %s',
                    (status, recommendation_id)
                )
        else:
            cursor.execute(
                'UPDATE trading_recommendations SET status = %s WHERE id = %s',
                (status, recommendation_id)
            )
        
        mysql.connection.commit()
        return True
    except Exception as e:
        print(f"Error updating recommendation: {e}")
        return False

@db_retry(max_attempts=3, delay=1)
def add_to_watchlist(user_id, symbol, company_name, notes=''):
    """Add stock to watchlist"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('''
            INSERT IGNORE INTO watchlists (user_id, symbol, company_name, notes)
            VALUES (%s, %s, %s, %s)
        ''', (user_id, symbol, company_name, notes))
        mysql.connection.commit()
        return True
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False

@db_retry(max_attempts=3, delay=1)
def get_watchlist(user_id):
    """Get user's watchlist"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM watchlists WHERE user_id = %s ORDER BY added_at DESC',
            (user_id,)
        )
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting watchlist: {e}")
        return []

@db_retry(max_attempts=3, delay=1)
def remove_from_watchlist(watchlist_id, user_id):
    """Remove stock from watchlist"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'DELETE FROM watchlists WHERE id = %s AND user_id = %s',
            (watchlist_id, user_id)
        )
        mysql.connection.commit()
        return True
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False



# --- HELPER FUNCTIONS (Your existing plotting code) ---
def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_price_chart(analyzer):
    if analyzer.price_history.empty:
        return None
    
    df = analyzer.price_history.sort_index().copy()
    df['SMA20'] = df['close'].rolling(20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['SMA200'] = df['close'].rolling(200).mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(df.index, df['close'], label='Giá đóng cửa', color='#1f77b4', linewidth=1.5)
    ax1.plot(df.index, df['SMA20'], label='SMA20', linestyle='--', color='#ff7f0e', alpha=0.8, linewidth=1.2)
    ax1.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='#2ca02c', alpha=0.8, linewidth=1.2)
    ax1.plot(df.index, df['SMA200'], label='SMA200', linestyle='--', color='#d62728', alpha=0.7, linewidth=1.2)
    
    ax1.set_title(f'LỊCH SỬ GIÁ {analyzer.symbol} - 1 NĂM GẦN NHẤT', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Giá (VNĐ)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' for i in range(len(df))]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
    ax2.set_ylabel('Khối lượng', fontsize=11)
    ax2.set_xlabel('Thời gian', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return plot_to_base64(fig)

def generate_industry_charts(analyzer):
    """Generate industry-specific financial charts"""
    charts = []
    if analyzer.final_metrics is None or analyzer.final_metrics.empty:
        return charts
    
    industry = analyzer.profile_info.get('industry', '').lower()
    industry2 = analyzer.profile_info.get('industry2', '').lower()
    plot_data = analyzer.final_metrics.tail(8)

    # BANKING
    if 'ngân hàng' in industry or 'bank' in industry:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{analyzer.symbol} - PHÂN TÍCH NGÂN HÀNG', fontsize=14, fontweight='bold')
        
        if 'LDR (%)' in plot_data.columns:
            axes[0, 0].plot(plot_data.index, plot_data['LDR (%)'], marker='o', color='#1f77b4', linewidth=2)
            axes[0, 0].axhline(85, color='green', linestyle='--', alpha=0.5, label='Ngưỡng an toàn 85%')
            axes[0, 0].set_title('Tỷ lệ Cho vay/Huy động (LDR)')
            axes[0, 0].set_ylabel('LDR (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'NIM (%)' in plot_data.columns:
            axes[0, 1].plot(plot_data.index, plot_data['NIM (%)'], marker='s', color='#ff7f0e', linewidth=2)
            axes[0, 1].set_title('Biên lãi suất ròng (NIM)')
            axes[0, 1].set_ylabel('NIM (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'CIR (%)' in plot_data.columns:
            axes[1, 0].plot(plot_data.index, plot_data['CIR (%)'], marker='^', color='#2ca02c', linewidth=2)
            axes[1, 0].axhline(40, color='red', linestyle='--', alpha=0.5, label='Ngưỡng hiệu quả 40%')
            axes[1, 0].set_title('Tỷ lệ Chi phí/Thu nhập (CIR)')
            axes[1, 0].set_ylabel('CIR (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'ROE (Quý) (%)' in plot_data.columns:
            axes[1, 1].plot(plot_data.index, plot_data['ROE (Quý) (%)'], marker='D', color='#d62728', linewidth=2)
            axes[1, 1].set_title('ROE theo Quý')
            axes[1, 1].set_ylabel('ROE (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts.append(('banking', plot_to_base64(fig)))
    
    # REAL ESTATE
    elif 'bất động sản' in industry:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{analyzer.symbol} - PHÂN TÍCH BẤT ĐỘNG SẢN', fontsize=14, fontweight='bold')
        
        if 'Hàng tồn kho (Tỷ)' in plot_data.columns:
            axes[0, 0].bar(range(len(plot_data)), plot_data['Hàng tồn kho (Tỷ)'], color='#1f77b4', alpha=0.7)
            axes[0, 0].set_title('Hàng tồn kho (Dự án BĐS)')
            axes[0, 0].set_ylabel('Tỷ đồng')
            axes[0, 0].set_xticks(range(len(plot_data)))
            axes[0, 0].set_xticklabels(plot_data.index, rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        if 'Người mua trả trước (Tỷ)' in plot_data.columns:
            axes[0, 1].bar(range(len(plot_data)), plot_data['Người mua trả trước (Tỷ)'], color='#ff7f0e', alpha=0.7)
            axes[0, 1].set_title('Người mua trả tiền trước')
            axes[0, 1].set_ylabel('Tỷ đồng')
            axes[0, 1].set_xticks(range(len(plot_data)))
            axes[0, 1].set_xticklabels(plot_data.index, rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        if 'Tỷ lệ Trả trước/Tồn kho (%)' in plot_data.columns:
            axes[1, 0].plot(plot_data.index, plot_data['Tỷ lệ Trả trước/Tồn kho (%)'], marker='o', color='#2ca02c', linewidth=2)
            axes[1, 0].set_title('Tỷ lệ Trả trước/Tồn kho')
            axes[1, 0].set_ylabel('%')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'FCF (Tỷ)' in plot_data.columns:
            colors = ['green' if x >= 0 else 'red' for x in plot_data['FCF (Tỷ)']]
            axes[1, 1].bar(range(len(plot_data)), plot_data['FCF (Tỷ)'], color=colors, alpha=0.7)
            axes[1, 1].axhline(0, color='black', linewidth=1)
            axes[1, 1].set_title('Dòng tiền tự do (FCF)')
            axes[1, 1].set_ylabel('Tỷ đồng')
            axes[1, 1].set_xticks(range(len(plot_data)))
            axes[1, 1].set_xticklabels(plot_data.index, rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        charts.append(('realestate', plot_to_base64(fig)))
    
    # INSURANCE
    elif 'bảo hiểm' in industry:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{analyzer.symbol} - PHÂN TÍCH BẢO HIỂM', fontsize=14, fontweight='bold')
        
        if 'Đầu tư/Tổng TS (%)' in plot_data.columns:
            axes[0, 0].plot(plot_data.index, plot_data['Đầu tư/Tổng TS (%)'], marker='o', color='#1f77b4', linewidth=2)
            axes[0, 0].set_title('Tỷ trọng Đầu tư/Tổng tài sản')
            axes[0, 0].set_ylabel('%')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'Lợi suất đầu tư (%)' in plot_data.columns:
            axes[0, 1].plot(plot_data.index, plot_data['Lợi suất đầu tư (%)'], marker='s', color='#ff7f0e', linewidth=2)
            axes[0, 1].set_title('Lợi suất đầu tư')
            axes[0, 1].set_ylabel('%')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'Tỷ lệ chi phí HĐ (%)' in plot_data.columns:
            axes[1, 0].plot(plot_data.index, plot_data['Tỷ lệ chi phí HĐ (%)'], marker='^', color='#2ca02c', linewidth=2)
            axes[1, 0].set_title('Tỷ lệ chi phí hoạt động')
            axes[1, 0].set_ylabel('%')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'Đòn bẩy tài chính (Lần)' in plot_data.columns:
            axes[1, 1].plot(plot_data.index, plot_data['Đòn bẩy tài chính (Lần)'], marker='D', color='#d62728', linewidth=2)
            axes[1, 1].set_title('Đòn bẩy tài chính')
            axes[1, 1].set_ylabel('Lần')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        charts.append(('insurance', plot_to_base64(fig)))
    
    # ENERGY (Oil & Gas, Power)
    elif 'dầu khí' in industry2 or 'điện' in industry or 'năng lượng' in industry or 'năng lượng' in industry2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{analyzer.symbol} - PHÂN TÍCH NĂNG LƯỢNG', fontsize=14, fontweight='bold')
        
        if 'EBITDA (Tỷ)' in plot_data.columns:
            axes[0, 0].bar(range(len(plot_data)), plot_data['EBITDA (Tỷ)'], color='#1f77b4', alpha=0.7)
            axes[0, 0].set_title('EBITDA')
            axes[0, 0].set_ylabel('Tỷ đồng')
            axes[0, 0].set_xticks(range(len(plot_data)))
            axes[0, 0].set_xticklabels(plot_data.index, rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        if 'Biên EBITDA (%)' in plot_data.columns:
            axes[0, 1].plot(plot_data.index, plot_data['Biên EBITDA (%)'], marker='o', color='#ff7f0e', linewidth=2)
            axes[0, 1].set_title('Biên EBITDA')
            axes[0, 1].set_ylabel('%')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'Vay ròng/EBITDA (Lần)' in plot_data.columns:
            axes[1, 0].plot(plot_data.index, plot_data['Vay ròng/EBITDA (Lần)'], marker='s', color='#2ca02c', linewidth=2)
            axes[1, 0].axhline(3, color='red', linestyle='--', alpha=0.5, label='Ngưỡng an toàn 3x')
            axes[1, 0].set_title('Vay ròng/EBITDA')
            axes[1, 0].set_ylabel('Lần')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'FCF (Tỷ)' in plot_data.columns:
            colors = ['green' if x >= 0 else 'red' for x in plot_data['FCF (Tỷ)']]
            axes[1, 1].bar(range(len(plot_data)), plot_data['FCF (Tỷ)'], color=colors, alpha=0.7)
            axes[1, 1].axhline(0, color='black', linewidth=1)
            axes[1, 1].set_title('Dòng tiền tự do (FCF)')
            axes[1, 1].set_ylabel('Tỷ đồng')
            axes[1, 1].set_xticks(range(len(plot_data)))
            axes[1, 1].set_xticklabels(plot_data.index, rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        charts.append(('energy', plot_to_base64(fig)))
    
    return charts

def generate_ownership_chart(analyzer):
    # ... (Your existing code for ownership chart) ...
    df_sh = analyzer.profile_info.get('shareholders')
    if df_sh is None or df_sh.empty:
        return None
    try:
        df_plot = df_sh.copy().sort_values('share_own_percent', ascending=False)
        df_plot['share_own_percent'] = df_plot['share_own_percent'].fillna(0)
        if df_plot['share_own_percent'].max() <= 1.0:
            df_plot['share_own_percent'] *= 100
        
        top_5 = df_plot.head(5)
        others = max(0, 100 - top_5['share_own_percent'].sum())
        sizes = list(top_5['share_own_percent']) + ([others] if others > 0 else [])
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90)
        plt.title(f'CƠ CẤU SỞ HỮU - {analyzer.symbol}', fontweight='bold')
        plt.tight_layout()
        return plot_to_base64(fig)
    except:
        return None

def generate_trading_signals_chart(trading_analyzer):
    """Generate trading signals visualization"""
    try:
        df = trading_analyzer.df.sort_index().copy()
        if len(df) < 50:
            return None
        
        # Calculate indicators
        df['SMA20'] = df['close'].rolling(20).mean()
        df['SMA50'] = df['close'].rolling(50).mean()
        df['SMA200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = sma + (2 * std)
        df['BB_Lower'] = sma - (2 * std)
        
        # Use last 100 days
        plot_data = df.tail(100)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Price with SMAs and Bollinger Bands
        ax1 = axes[0]
        ax1.plot(plot_data.index, plot_data['close'], label='Giá đóng cửa', color='#000000', linewidth=2)
        ax1.plot(plot_data.index, plot_data['SMA20'], label='SMA20', linestyle='--', color='#ff7f0e', alpha=0.8)
        ax1.plot(plot_data.index, plot_data['SMA50'], label='SMA50', linestyle='--', color='#2ca02c', alpha=0.8)
        ax1.plot(plot_data.index, plot_data['SMA200'], label='SMA200', linestyle='--', color='#d62728', alpha=0.7)
        ax1.fill_between(plot_data.index, plot_data['BB_Upper'], plot_data['BB_Lower'], alpha=0.1, color='#1f77b4')
        ax1.set_title('Giá + SMA + Bollinger Bands', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Giá (VNĐ)', fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # RSI
        ax2 = axes[1]
        ax2.plot(plot_data.index, plot_data['RSI'], label='RSI(14)', color='#1f77b4', linewidth=2)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.fill_between(plot_data.index, 30, 70, alpha=0.1, color='#gray')
        ax2.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        # MACD
        ax3 = axes[2]
        ax3.plot(plot_data.index, plot_data['MACD'], label='MACD', color='#1f77b4', linewidth=2)
        ax3.plot(plot_data.index, plot_data['MACD_Signal'], label='Signal Line', color='#ff7f0e', linewidth=2)
        ax3.bar(plot_data.index, plot_data['MACD'] - plot_data['MACD_Signal'], label='Histogram', 
                color=['#2ca02c' if x >= 0 else '#d62728' for x in (plot_data['MACD'] - plot_data['MACD_Signal'])], alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax3.set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('MACD', fontsize=10)
        ax3.set_xlabel('Thời gian', fontsize=10)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return plot_to_base64(fig)
    except Exception as e:
        print(f"Error generating trading signals chart: {e}")
        return None


# --- SEARCH HISTORY FUNCTIONS ---
@db_retry(max_attempts=3, delay=1)
def add_to_search_history(user_id, symbol, company_name):
    """Add a stock search to user's history"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        # Check if this search already exists recently (within last hour)
        cursor.execute(
            'SELECT id FROM search_history WHERE user_id = %s AND symbol = %s AND searched_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)',
            (user_id, symbol)
        )
        existing = cursor.fetchone()
        
        if not existing:
            # Insert new search
            cursor.execute(
                'INSERT INTO search_history (user_id, symbol, company_name) VALUES (%s, %s, %s)',
                (user_id, symbol, company_name)
            )
            mysql.connection.commit()
    except Exception as e:
        print(f"Error adding to search history: {e}")

@db_retry(max_attempts=3, delay=1)
def get_user_search_history(user_id, limit=10):
    """Get user's recent search history"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            '''SELECT DISTINCT symbol, company_name, MAX(searched_at) as last_searched 
               FROM search_history 
               WHERE user_id = %s 
               GROUP BY symbol, company_name
               ORDER BY last_searched DESC 
               LIMIT %s''',
            (user_id, limit)
        )
        return cursor.fetchall()
    except Exception as e:
        print(f"Error getting search history: {e}")
        return []

def clear_user_search_history(user_id):
    """Clear all search history for a user"""
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('DELETE FROM search_history WHERE user_id = %s', (user_id,))
        mysql.connection.commit()
        return True
    except Exception as e:
        print(f"Error clearing search history: {e}")
        return False

# --- NEW ROUTES (The "Gatekeeper" Logic) ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        # Check Account in MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        
        if account:
            # Verify hashed password
            if verify_password(password, account['password']):
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                return redirect(url_for('index'))
            else:
                msg = 'Sai tên đăng nhập hoặc mật khẩu!'
        else:
            msg = 'Sai tên đăng nhập hoặc mật khẩu!'
            
    return render_template('login.html', msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle new user registration (Optional)"""
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        password_confirm = request.form.get('password_confirm', '')
        
        # Validate password
        if len(password) < 6:
            msg = 'Mật khẩu phải có ít nhất 6 ký tự!'
        elif password != password_confirm:
            msg = 'Mật khẩu không khớp!'
        else:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
            account = cursor.fetchone()
            
            if account:
                msg = 'Tài khoản đã tồn tại!'
            else:
                # Hash password before storing
                hashed_password = hash_password(password)
                cursor.execute(
                    'INSERT INTO users (username, password, email) VALUES (%s, %s, NULL)',
                    (username, hashed_password)
                )
                mysql.connection.commit()
                msg = 'Đăng ký thành công! Vui lòng đăng nhập.'
                return redirect(url_for('login'))
            
    return render_template('register.html', msg=msg)

@app.route('/logout')
def logout():
    """Clear session"""
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# --- EXISTING ROUTES (Protected by the Gatekeeper) ---

@app.route('/')
def index():
    """Home page with search form"""
    # [GATEKEEPER CHECK]
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    # Get user's search history
    search_history = get_user_search_history(session['id'], limit=10)
        
    return render_template('index.html', username=session['username'], search_history=search_history)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze stock and show results"""
    # [GATEKEEPER CHECK]
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip().upper()
    else:
        symbol = request.args.get('symbol', '').strip().upper()
    
    if not symbol:
        return render_template('index.html', error='Vui lòng nhập mã cổ phiếu', username=session['username'], search_history=get_user_search_history(session['id']))
    
    try:
        # Create analyzer
        analyzer = BusinessAnalyzer(symbol)
        
        # Load data
        if not analyzer.get_company_info():
            return render_template('index.html', error=f'Không thể tải thông tin cho mã {symbol}', username=session['username'], search_history=get_user_search_history(session['id']))
        
        analyzer.get_historical_price()
        
        if analyzer.get_financial_data():
            analyzer.calculate_metrics()
        
        # Prepare company info early (needed for trading signals)
        info = analyzer.profile_info
        m_cap = info.get('market_cap', 0)
        cap_str = f"{m_cap/1_000_000_000:,.0f} Tỷ VNĐ" if m_cap > 0 else "N/A"
        company_name = info.get('organ_name', '') or info.get('short_name', symbol)
        
        # Generate charts
        price_chart = generate_price_chart(analyzer)
        # Note: I'm calling your original function name here
        industry_charts = generate_industry_charts(analyzer) 
        ownership_chart = generate_ownership_chart(analyzer)
        
        # --- NEW: TRADING SIGNALS ANALYSIS ---
        trading_signals = None
        trading_chart = None
        if not analyzer.price_history.empty:
            trading_analyzer = TradingSignalAnalyzer(analyzer.price_history)
            trading_signals = trading_analyzer.generate_comprehensive_signal()
            trading_chart = generate_trading_signals_chart(trading_analyzer)
            
            # Save trading signal to database
            save_trading_signal(session['id'], symbol, company_name, trading_signals)
            
            # Auto-generate recommendation based on signal (create for any signal type)
            if trading_signals:
                confidence = int(trading_signals.get('final_confidence', 0.5) * 100)
                current_price = info.get('price', 0)
                signal_action = trading_signals.get('final_signal', 'HOLD')
                reason = f"Based on technical analysis: {signal_action} signal"
                
                # Calculate potential targets based on volatility
                if signal_action == 'BUY':
                    target_multiplier = 1.05
                    stop_loss_multiplier = 0.97
                elif signal_action == 'SELL':
                    target_multiplier = 0.95
                    stop_loss_multiplier = 1.03
                else:  # HOLD
                    target_multiplier = 1.02
                    stop_loss_multiplier = 0.98
                
                target_price = current_price * target_multiplier if current_price > 0 else None
                stop_loss = current_price * stop_loss_multiplier if current_price > 0 else None
                
                save_recommendation(
                    session['id'], symbol, company_name,
                    action=signal_action,
                    confidence=confidence,
                    reason=reason,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss
                )
        
        # Prepare metrics table
        metrics_html = None
        if analyzer.final_metrics is not None and not analyzer.final_metrics.empty:
            valid_cols = analyzer.final_metrics.columns[
                (analyzer.final_metrics != 0).any() & (analyzer.final_metrics.notna().any())
            ]
            metrics_df = analyzer.final_metrics[valid_cols].tail(8)
            metrics_html = metrics_df.T.to_html(classes='table table-striped table-hover', 
                                              float_format=lambda x: f'{x:,.2f}')
        
        # Prepare price history table
        price_history_html = None
        if not analyzer.price_history.empty:
            df = analyzer.price_history.sort_index(ascending=False).head(30).copy()
            display_df = pd.DataFrame({
                'Ngày': df.index.strftime('%d/%m/%Y'),
                'Mở cửa': df['open'].apply(lambda x: f"{x:,.2f}"),
                'Cao nhất': df['high'].apply(lambda x: f"{x:,.2f}"),
                'Thấp nhất': df['low'].apply(lambda x: f"{x:,.2f}"),
                'Đóng cửa': df['close'].apply(lambda x: f"{x:,.2f}"),
                'Khối lượng': df['volume'].apply(lambda x: f"{int(x):,}")
            })
            price_history_html = display_df.to_html(classes='table table-striped table-sm', index=False)
        
        # --- STRATEGY EVALUATION ---
        strategy_eval = None
        if not analyzer.price_history.empty:
            try:
                price_dict = {symbol: analyzer.price_history[['open','high','low','close','volume']].copy()}
                summary_df, details = evaluate_universe(price_dict, initial_capital=1_000_000, risk_per_trade=0.01)
                
                perf = summary_df.loc[symbol].to_dict()
                detail = details.get(symbol, {})
                equity_img = None
                
                try:
                    eq_df = detail.get('equity_curve')
                    if eq_df is not None and not eq_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(eq_df.index, eq_df.values, linewidth=2, color='#2E86AB')
                        ax.set_title(f'Strategy Equity Curve - {symbol}', fontweight='bold', fontsize=12)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Portfolio Value (VND)')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        img_io = io.BytesIO()
                        fig.savefig(img_io, format='png', dpi=100)
                        img_io.seek(0)
                        equity_img = base64.b64encode(img_io.getvalue()).decode()
                        plt.close(fig)
                except Exception as e:
                    print(f"Error generating equity curve: {e}")
                
                strategy_eval = {
                    'symbol': symbol,
                    'perf': perf,
                    'trades': detail.get('trades', []),
                    'equity_img': equity_img
                }
            except Exception as e:
                print(f"Error evaluating strategy: {e}")
        
        
        # Build company info dictionary
        company_info = {
            'symbol': symbol,
            'name': company_name,
            'industry': info.get('industry', 'N/A'),
            'industry2': info.get('industry2', ''),
            'exchange': info.get('exchange', 'N/A'),
            'website': info.get('website', ''),
            'established_year': info.get('established_year', ''),
            'employees': info.get('no_employees', 0),
            'shareholders_count': info.get('no_shareholders', 0),
            'foreign_percent': info.get('foreign_percent', 0),
            'outstanding_share': info.get('outstanding_share', 0),
            'charter_capital': info.get('charter_capital', 0),
            'price': info.get('price', 0),
            'pct_change': info.get('pct_change', 0),
            'market_cap': cap_str,
            'officers': info.get('officers', [])[:5],
            'news': info.get('news', []),
            'events': info.get('events', [])
        }
        
        # Save to search history
        add_to_search_history(session['id'], symbol, company_name)
        
        return render_template('results.html',
                             username=session['username'],
                             company_info=company_info,
                             price_chart=price_chart,
                             industry_charts=industry_charts,
                             ownership_chart=ownership_chart,
                             trading_signals=trading_signals,
                             trading_chart=trading_chart,
                             metrics_html=metrics_html,
                             price_history_html=price_history_html,
                             strategy_eval=strategy_eval)
    
    except Exception as e:
        return render_template('index.html', error=f'Lỗi khi phân tích {symbol}: {str(e)}', username=session['username'], search_history=get_user_search_history(session['id']))


@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear user's search history"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    clear_user_search_history(session['id'])
    flash('Lịch sử tra cứu đã được xóa!', 'success')
    return redirect(url_for('index'))

@app.route('/signals/<symbol>')
def view_signals(symbol):
    """View trading signal history for a stock"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    signals_history = get_trading_signals_history(session['id'], symbol.upper())
    
    # Get company name from first signal
    company_name = signals_history[0]['company_name'] if signals_history else symbol
    
    return render_template('signals_history.html',
                         username=session['username'],
                         symbol=symbol.upper(),
                         company_name=company_name,
                         signals_history=signals_history)

@app.route('/recommendations')
def view_recommendations():
    """View all active trading recommendations"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    recommendations = get_user_recommendations(session['id'], status='ACTIVE')
    completed = get_user_recommendations(session['id'], status='COMPLETED')
    
    return render_template('recommendations.html',
                         username=session['username'],
                         active_recommendations=recommendations,
                         completed_recommendations=completed)

@app.route('/watchlist')
def view_watchlist():
    """View user's watchlist"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    watchlist = get_watchlist(session['id'])
    
    return render_template('watchlist.html',
                         username=session['username'],
                         watchlist=watchlist)

@app.route('/add-to-watchlist', methods=['POST'])
def add_watchlist():
    """Add stock to watchlist"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    symbol = request.form.get('symbol', '').upper()
    company_name = request.form.get('company_name', symbol)
    notes = request.form.get('notes', '')
    
    if add_to_watchlist(session['id'], symbol, company_name, notes):
        flash(f'{symbol} đã được thêm vào danh sách theo dõi!', 'success')
    else:
        flash(f'{symbol} có thể đã tồn tại trong danh sách!', 'warning')
    
    return redirect(request.referrer or url_for('index'))

@app.route('/update-recommendation/<int:rec_id>/<status>', methods=['POST'])
def update_rec_status(rec_id, status):
    """Update recommendation status"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    realized_price = request.form.get('realized_price')
    realized_price = float(realized_price) if realized_price else None
    
    if update_recommendation_status(rec_id, status.upper(), realized_price):
        flash('Cập nhật khuyến nghị thành công!', 'success')
    else:
        flash('Lỗi khi cập nhật khuyến nghị!', 'danger')
    
    return redirect(request.referrer or url_for('view_recommendations'))

@app.route('/remove-watchlist/<int:watchlist_id>', methods=['POST'])
def remove_watchlist(watchlist_id):
    """Remove stock from watchlist"""
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    
    if remove_from_watchlist(watchlist_id, session['id']):
        flash('Đã xóa khỏi danh sách theo dõi!', 'success')
    else:
        flash('Lỗi khi xóa khỏi danh sách!', 'danger')
    
    return redirect(request.referrer or url_for('view_watchlist'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)