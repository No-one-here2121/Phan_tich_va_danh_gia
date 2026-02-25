#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta

# Import classes
from demo_sp import BusinessAnalyzer
from trading_signals import TradingSignalAnalyzer

app = Flask(__name__)
app.secret_key = 'test-key'

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=80, bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url


def generate_price_chart(df_price):
    """Generate price chart with moving averages"""
    try:
        if df_price is None or df_price.empty or len(df_price) < 2:
            return None
        
        df_copy = df_price.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.sort_values('date')
        
        df_copy['SMA_20'] = df_copy['close'].rolling(window=20, min_periods=1).mean()
        df_copy['SMA_50'] = df_copy['close'].rolling(window=50, min_periods=1).mean()
        df_copy['SMA_200'] = df_copy['close'].rolling(window=200, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_copy['date'], df_copy['close'], label='Giá đóng cửa', color='black', linewidth=2)
        ax.plot(df_copy['date'], df_copy['SMA_20'], label='SMA 20', alpha=0.7, linewidth=1)
        ax.plot(df_copy['date'], df_copy['SMA_50'], label='SMA 50', alpha=0.7, linewidth=1)
        ax.plot(df_copy['date'], df_copy['SMA_200'], label='SMA 200', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá (đ)')
        ax.set_title('Biểu đồ Giá Cổ Phiếu (1 Năm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plot_to_base64(fig)
    except Exception as e:
        print(f"Error generating price chart: {e}")
        return None


def generate_trading_signals_chart(df_price):
    """Generate chart with trading signals"""
    try:
        if df_price is None or df_price.empty or len(df_price) < 50:
            return None
        
        df_copy = df_price.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.sort_values('date')
        
        analyzer = TradingSignalAnalyzer(df_copy)
        signals = analyzer.generate_signals()
        
        if signals is None or signals.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Price with signals
        ax1.plot(df_copy['date'], df_copy['close'], label='Giá đóng cửa', color='black', linewidth=2)
        
        buy_signals = signals[signals['Signal'] == 'BUY']
        sell_signals = signals[signals['Signal'] == 'SELL']
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals['date'], buy_signals['close'], color='green', marker='^', s=100, label='BUY')
        if not sell_signals.empty:
            ax1.scatter(sell_signals['date'], sell_signals['close'], color='red', marker='v', s=100, label='SELL')
        
        ax1.set_ylabel('Giá (đ)')
        ax1.set_title('Tín Hiệu Giao Dịch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI
        ax2.plot(signals['date'], signals['RSI'], label='RSI', color='blue', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.set_xlabel('Ngày')
        ax2.set_ylabel('RSI')
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return plot_to_base64(fig)
    except Exception as e:
        print(f"Error generating trading signals chart: {e}")
        return None


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', username='Test User')


@app.route('/analyze')
def analyze():
    """Stock analysis page"""
    symbol = request.args.get('symbol', 'VND').upper()
    error = None
    company_info = {}
    price_chart = None
    ownership_chart = None
    trading_chart = None
    metrics_html = None
    price_history_html = None
    trading_signals = {}
    
    try:
        print(f"\n[ANALYZE] Phân tích {symbol}...")
        
        analyzer = BusinessAnalyzer(symbol)
        
        if not analyzer.get_company_info():
            print(f"Cảnh báo: Không tải được company info, tiếp tục...")
        
        analyzer.get_historical_price()
        analyzer.get_financial_data()
        analyzer.calculate_metrics()
        
        # Get data
        company_info = {
            'symbol': symbol,
            'organ_name': analyzer.profile_info.get('organ_name', ''),
            'short_name': analyzer.profile_info.get('short_name', ''),
            'industry': analyzer.profile_info.get('industry', ''),
            'price': analyzer.profile_info.get('price', 0),
            'pct_change': analyzer.profile_info.get('pct_change', 0),
            'market_cap': analyzer.profile_info.get('market_cap', 0),
            'employees': analyzer.profile_info.get('no_employees', 0),
            'shareholders_count': analyzer.profile_info.get('no_shareholders', 0),
            'foreign_percent': analyzer.profile_info.get('foreign_percent', 0),
            'outstanding_share': analyzer.profile_info.get('outstanding_share', 0),
            'charter_capital': analyzer.profile_info.get('charter_capital', 0),
            'price': analyzer.profile_info.get('price', 0),
            'pct_change': analyzer.profile_info.get('pct_change', 0),
            'market_cap': analyzer.profile_info.get('market_cap', 0),
            'officers': analyzer.profile_info.get('officers', [])[:5],
            'news': analyzer.profile_info.get('news', []),
            'events': analyzer.profile_info.get('events', [])
        }
        
        # Charts
        price_chart = generate_price_chart(analyzer.historical_price)
        trading_chart = generate_trading_signals_chart(analyzer.historical_price)
        
        # Trading signals
        if analyzer.historical_price is not None and not analyzer.historical_price.empty:
            signal_analyzer = TradingSignalAnalyzer(analyzer.historical_price)
            signals = signal_analyzer.generate_signals()
            if signals is not None and not signals.empty:
                latest = signals.iloc[-1]
                trading_signals = {
                    'signal': latest.get('Signal', 'HOLD'),
                    'rsi': round(latest.get('RSI', 50), 2),
                    'macd': round(latest.get('MACD', 0), 2),
                    'signal_line': round(latest.get('Signal_Line', 0), 2)
                }
        
        return render_template('results.html',
                             username='Test User',
                             company_info=company_info,
                             price_chart=price_chart,
                             industry_charts=[],
                             ownership_chart=ownership_chart,
                             trading_signals=trading_signals,
                             trading_chart=trading_chart,
                             metrics_html=metrics_html,
                             price_history_html=price_history_html,
                             strategy_eval=None)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        error = f'Lỗi khi phân tích {symbol}: {str(e)}'
    
    return render_template('index.html', error=error, username='Test User')


@app.route('/ai-dudoan', methods=['GET', 'POST'])
def ai_predict():
    """AI Prediction page"""
    prediction_result = None
    ticker = None
    error = None
    
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        
        if not ticker:
            error = 'Vui lòng nhập mã cổ phiếu'
        else:
            try:
                print(f"\n[AI PREDICT] Dự đoán cho {ticker}...")
                
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.preprocessing import StandardScaler
                
                # Get financial data
                analyzer = BusinessAnalyzer(ticker)
                
                # Try to get data - don't fail on company info
                try:
                    analyzer.get_company_info()
                except Exception as e:
                    print(f"Cảnh báo: Không tải được company info - {e}")
                    pass
                
                analyzer.get_historical_price()
                analyzer.get_financial_data()
                analyzer.calculate_metrics()
                
                # Prepare data for prediction
                finance = analyzer.finance
                income_stmt = finance.income_statement(lang='vi', dropna=True)
                balance_stmt = finance.balance_sheet(lang='vi', dropna=True)
                
                # Find columns
                equity_col = None
                net_income_col = None
                revenue_col = None
                
                for col in balance_stmt.columns:
                    if 'vốn chủ sở hữu' in str(col).lower():
                        equity_col = col
                        break
                
                for col in income_stmt.columns:
                    if 'lợi nhuận sau thuế' in str(col).lower() and 'công ty mẹ' in str(col).lower():
                        net_income_col = col
                        break
                
                for col in income_stmt.columns:
                    if 'doanh thu' in str(col).lower():
                        revenue_col = col
                        break
                
                if not (revenue_col and net_income_col and equity_col):
                    error = f'Không tìm thấy dữ liệu tài chính đủ cho {ticker}'
                else:
                    # Prepare quarterly data
                    income_stmt['Period'] = income_stmt['Năm'].astype(str) + 'Q' + income_stmt['Kỳ'].astype(str)
                    balance_stmt['Period'] = balance_stmt['Năm'].astype(str) + 'Q' + balance_stmt['Kỳ'].astype(str)
                    
                    quarterly_data = income_stmt[['Period', revenue_col, net_income_col]].copy()
                    quarterly_data.rename(columns={
                        revenue_col: 'Doanh thu',
                        net_income_col: 'Lợi nhuận ròng'
                    }, inplace=True)
                    
                    # Merge with equity
                    balance_equity = balance_stmt[['Period', equity_col]].copy()
                    balance_equity.rename(columns={equity_col: 'Vốn chủ sở hữu'}, inplace=True)
                    quarterly_data = quarterly_data.merge(balance_equity, on='Period', how='left')
                    quarterly_data = quarterly_data.dropna()
                    
                    # Convert to numeric
                    quarterly_data['Doanh thu'] = pd.to_numeric(quarterly_data['Doanh thu'], errors='coerce')
                    quarterly_data['Lợi nhuận ròng'] = pd.to_numeric(quarterly_data['Lợi nhuận ròng'], errors='coerce')
                    quarterly_data['Vốn chủ sở hữu'] = pd.to_numeric(quarterly_data['Vốn chủ sở hữu'], errors='coerce')
                    
                    quarterly_data = quarterly_data.dropna()
                    
                    if len(quarterly_data) < 2:
                        error = f'Dữ liệu không đủ để dự đoán (cần ít nhất 2 quý)'
                    else:
                        # Scale data
                        quarterly_data['Doanh thu_B'] = quarterly_data['Doanh thu'] / 1e9
                        quarterly_data['Lợi nhuận_B'] = quarterly_data['Lợi nhuận ròng'] / 1e9
                        
                        # Calculate QoQ change
                        quarterly_data['Doanh thu QoQ (%)'] = quarterly_data['Doanh thu'].pct_change() * 100
                        quarterly_data['Lợi nhuận QoQ (%)'] = quarterly_data['Lợi nhuận ròng'].pct_change() * 100
                        quarterly_data['Doanh thu QoQ (%)'].fillna(0, inplace=True)
                        quarterly_data['Lợi nhuận QoQ (%)'].fillna(0, inplace=True)
                        
                        # Calculate ROE
                        quarterly_data['ROE (%)'] = (quarterly_data['Lợi nhuận ròng'] / quarterly_data['Vốn chủ sở hữu']) * 100
                        
                        # Features for model
                        features_list = ['Doanh thu_B', 'Lợi nhuận_B', 'Doanh thu QoQ (%)', 'Lợi nhuận QoQ (%)']
                        has_roe = False
                        if 'ROE (%)' in quarterly_data.columns and quarterly_data['ROE (%)'].notna().any():
                            features_list.append('ROE (%)')
                            has_roe = True
                        
                        X = quarterly_data[features_list].values
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Train models
                        y_revenue = quarterly_data['Doanh thu_B'].shift(-1).dropna()
                        X_revenue = X_scaled[:-1]
                        
                        y_profit = quarterly_data['Lợi nhuận_B'].shift(-1).dropna()
                        X_profit = X_scaled[:-1]
                        
                        if len(X_revenue) > 1 and len(y_revenue) > 1:
                            model_revenue = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                            model_revenue.fit(X_revenue, y_revenue)
                            
                            model_profit = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                            model_profit.fit(X_profit, y_profit)
                            
                            has_roe_model = False
                            pred_roe = 0
                            roe_change = 0
                            
                            if has_roe:
                                y_roe = quarterly_data['ROE (%)'].shift(-1).dropna()
                                X_roe = X_scaled[:-1]
                                if len(X_roe) > 1 and len(y_roe) > 1:
                                    model_roe = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                                    model_roe.fit(X_roe, y_roe)
                                    has_roe_model = True
                            
                            # Make predictions
                            X_latest = X_scaled[-1:]
                            pred_revenue = model_revenue.predict(X_latest)[0]
                            pred_profit = model_profit.predict(X_latest)[0]
                            
                            if has_roe_model:
                                pred_roe = model_roe.predict(X_latest)[0]
                            
                            # Get current values
                            current_revenue = quarterly_data.iloc[-1]['Doanh thu_B']
                            current_profit = quarterly_data.iloc[-1]['Lợi nhuận_B']
                            current_roe = quarterly_data.iloc[-1]['ROE (%)'] if 'ROE (%)' in quarterly_data.columns else 0
                            
                            # Calculate changes
                            revenue_change = ((pred_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
                            profit_change = ((pred_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
                            if has_roe_model:
                                roe_change = pred_roe - current_roe
                            
                            # Calculate next quarter
                            current_quarter = quarterly_data.iloc[-1]['Period']
                            year, quarter_str = current_quarter.split('Q')
                            year = int(year)
                            quarter = int(quarter_str)
                            
                            if quarter == 4:
                                next_quarter = f"{year + 1}Q1"
                            else:
                                next_quarter = f"{year}Q{quarter + 1}"
                            
                            # Calculate recommendation score
                            recommendation_score = 0
                            if revenue_change > 3:
                                recommendation_score += 1
                            if profit_change > 3:
                                recommendation_score += 1
                            if current_profit / current_revenue * 100 > 15:
                                recommendation_score += 1
                            if quarterly_data.tail(4)['Doanh thu QoQ (%)'].mean() > 5:
                                recommendation_score += 1
                            if quarterly_data.tail(4)['Doanh thu QoQ (%)'].std() < 15:
                                recommendation_score += 1
                            if revenue_change < -5:
                                recommendation_score -= 1
                            if profit_change < -5:
                                recommendation_score -= 1
                            
                            # Analysis summary
                            if recommendation_score >= 3:
                                analysis_summary = "✅ Doanh nghiệp có triển vọng tích cực. Đánh giá: CÓ THỂ NHỌ."
                            elif recommendation_score >= 1:
                                analysis_summary = "🟡 Doanh nghiệp có điểm tích cực nhưng còn không chắc chắn. Đánh giá: CẦN QUAN SÁT THÊM."
                            else:
                                analysis_summary = "🔴 Doanh nghiệp có dấu hiệu suy giảm. Đánh giá: CẦN THẬN TRỌNG."
                            
                            prediction_result = {
                                'ticker': ticker,
                                'current_quarter': current_quarter,
                                'next_quarter': next_quarter,
                                'revenue_current': current_revenue,
                                'revenue_pred': pred_revenue,
                                'revenue_change': revenue_change,
                                'profit_current': current_profit,
                                'profit_pred': pred_profit,
                                'profit_change': profit_change,
                                'has_roe': has_roe_model,
                                'roe_current': current_roe if has_roe_model else 0,
                                'roe_pred': pred_roe if has_roe_model else 0,
                                'roe_change': roe_change if has_roe_model else 0,
                                'recommendation_score': max(0, min(5, recommendation_score)),
                                'analysis_summary': analysis_summary
                            }
                        else:
                            error = 'Dữ liệu không đủ để huấn luyện mô hình'
                
            except Exception as e:
                error = f'Lỗi khi dự đoán: {str(e)}'
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    return render_template('AIdudoan.html',
                         prediction_result=prediction_result,
                         ticker=ticker,
                         error=error)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 CHẠY FLASK APP - PHIÊN BẢN ĐƠN GIẢN (KHÔNG CẦN MySQL)")
    print("="*60)
    print("\n📌 Truy cập tại: http://localhost:5000")
    print("📌 Dừng app: Ấn Ctrl+C")
    print("\n✨ Tính năng:")
    print("   • Trang chủ: http://localhost:5000/")
    print("   • Phân tích CP: Nhập mã -> /analyze")
    print("   • AI Dự đoán: http://localhost:5000/ai-dudoan")
    print("="*60 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)
