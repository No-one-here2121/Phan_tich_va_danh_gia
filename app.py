from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from datetime import datetime, timedelta
from demo_sp import BusinessAnalyzer
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_price_chart(analyzer):
    """Generate price history chart"""
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
    """Generate industry-specific charts"""
    charts = []
    
    if analyzer.final_metrics is None or analyzer.final_metrics.empty:
        return charts
    
    industry = analyzer.profile_info.get('industry', '').lower()
    industry2 = analyzer.profile_info.get('industry2', '').lower()
    plot_data = analyzer.final_metrics.tail(8)
    
    # Banking
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
        charts.append({'title': 'Phân tích ngân hàng', 'image': plot_to_base64(fig)})
    
    # Real Estate
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
        charts.append({'title': 'Phân tích bất động sản', 'image': plot_to_base64(fig)})
    
    # Add growth chart if available
    if 'Tăng trưởng DT (YoY %)' in plot_data.columns and 'Tăng trưởng LN (YoY %)' in plot_data.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_data[['Tăng trưởng DT (YoY %)', 'Tăng trưởng LN (YoY %)']].plot(kind='bar', ax=ax)
        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'{analyzer.symbol} - TĂNG TRƯỞNG YoY', fontsize=14, fontweight='bold')
        ax.set_ylabel('Tăng trưởng (%)')
        ax.set_xlabel('Kỳ')
        ax.legend(['Tăng trưởng DT (%)', 'Tăng trưởng LN (%)'])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        charts.append({'title': 'Tăng trưởng YoY', 'image': plot_to_base64(fig)})
    
    return charts

def generate_ownership_chart(analyzer):
    """Generate ownership pie chart"""
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
        labels = list(top_5['share_holder']) + (['Khác'] if others > 0 else [])
        sizes = list(top_5['share_own_percent']) + ([others] if others > 0 else [])
        
        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', 
                                           startangle=90, colors=sns.color_palette('pastel'), 
                                           pctdistance=0.85)
        ax.legend(wedges, labels, title="Cổ đông", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title(f'CƠ CẤU SỞ HỮU - {analyzer.symbol}', fontweight='bold')
        plt.tight_layout()
        
        return plot_to_base64(fig)
    except:
        return None

@app.route('/')
def index():
    """Home page with search form"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze stock and show results"""
    symbol = request.form.get('symbol', '').strip().upper()
    
    if not symbol:
        return render_template('index.html', error='Vui lòng nhập mã cổ phiếu')
    
    try:
        # Create analyzer
        analyzer = BusinessAnalyzer(symbol)
        
        # Load data
        if not analyzer.get_company_info():
            return render_template('index.html', error=f'Không thể tải thông tin cho mã {symbol}')
        
        analyzer.get_historical_price()
        
        if analyzer.get_financial_data():
            analyzer.calculate_metrics()
        
        # Generate charts
        price_chart = generate_price_chart(analyzer)
        industry_charts = generate_industry_charts(analyzer)
        ownership_chart = generate_ownership_chart(analyzer)
        
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
        
        # Prepare company info
        info = analyzer.profile_info
        m_cap = info.get('market_cap', 0)
        cap_str = f"{m_cap/1_000_000_000:,.0f} Tỷ VNĐ" if m_cap > 0 else "N/A"
        
        company_info = {
            'symbol': symbol,
            'name': info.get('organ_name', '') or info.get('short_name', ''),
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
        
        return render_template('results.html',
                             company_info=company_info,
                             price_chart=price_chart,
                             industry_charts=industry_charts,
                             ownership_chart=ownership_chart,
                             metrics_html=metrics_html,
                             price_history_html=price_history_html)
    
    except Exception as e:
        return render_template('index.html', error=f'Lỗi khi phân tích {symbol}: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
