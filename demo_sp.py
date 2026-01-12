import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime, timedelta
from vnstock import Finance, Company, Quote, Vnstock

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Cấu hình hiển thị pandas trong console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000) 
pd.options.display.float_format = '{:,.2f}'.format

class BusinessAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol.upper()
        self.company = Company(symbol=self.symbol, source='VCI')
        self.finance = Finance(symbol=self.symbol, source='VCI')
        self.quote = Quote(symbol=self.symbol, source='VCI')
        
        self.raw_reports = pd.DataFrame() 
        self.final_metrics = None
        self.price_history = pd.DataFrame()
        self.ratio_df = pd.DataFrame()
        self.profile_info = {
            'officers': [], 'subsidiaries': [], 'shareholders': pd.DataFrame(),
            'news': [], 'events': []
        } 

    def get_company_info(self):
        print(f"--- Đang tải thông tin {self.symbol} (Nguồn: VCI) ---")
        try:
            overview = self.company.overview()
            if not overview.empty:
                item = overview.iloc[0]
                self.profile_info['industry'] = item.get('icb_name3', '') 
                self.profile_info['exchange'] = item.get('exchange', 'VN')
            
            stats = self.company.trading_stats()
            if not stats.empty:
                item = stats.iloc[0]
                self.profile_info['price'] = item.get('close_price', 0)
                self.profile_info['pct_change'] = item.get('price_change_pct', 0)
                self.profile_info['market_cap'] = item.get('market_cap', 0)
            
            try: self.profile_info['officers'] = self.company.officers().head(10).to_dict('records')
            except: pass
            try: self.profile_info['subsidiaries'] = self.company.subsidiaries().head(10).to_dict('records')
            except: pass
            try: self.profile_info['shareholders'] = self.company.shareholders()
            except: pass
            
            try: 
                df_news = self.company.news()
                if not df_news.empty:
                    latest = df_news.head(10).copy()
                    if 'public_date' in latest.columns:
                        latest['date_str'] = pd.to_datetime(latest['public_date'], unit='ms').dt.strftime('%d/%m/%Y')
                    else: latest['date_str'] = 'N/A'
                    if 'news_source_link' not in latest.columns: latest['news_source_link'] = ''
                    self.profile_info['news'] = latest.to_dict('records')
            except: pass

            try:
                df_events = self.company.events()
                if not df_events.empty:
                    evt_process = df_events.copy()
                    date_col = next((c for c in ['exright_date', 'public_date', 'notify_date'] if c in evt_process.columns), None)
                    evt_process['sort_date'] = pd.to_datetime(evt_process[date_col], errors='coerce')
                    evt_process = evt_process.sort_values('sort_date', ascending=False).head(10)
                    evt_process['date_str'] = evt_process['sort_date'].dt.strftime('%d/%m/%Y').fillna('N/A')
                    if 'source_url' in evt_process.columns: evt_process['event_link'] = evt_process['source_url']
                    else: evt_process['event_link'] = ''
                    if 'event_title' in evt_process.columns: evt_process['display_name'] = evt_process['event_title']
                    elif 'event_list_name' in evt_process.columns: evt_process['display_name'] = evt_process['event_list_name']
                    else: evt_process['display_name'] = 'Sự kiện doanh nghiệp'
                    self.profile_info['events'] = evt_process.to_dict('records')
            except: pass
            return True
        except: return False

    def get_historical_price(self):
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            df = self.quote.history(start=start_date, end=end_date)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                self.price_history = df
                return True
        except: return False

    def get_financial_data(self):
        print(f"--- Đang tải báo cáo tài chính (Theo Quý)... ---")
        try:
            dfs = []
            for func in [self.finance.income_statement, self.finance.balance_sheet, self.finance.cash_flow]:
                try:
                    df = func(period='quarter', lang='vi', dropna=True)
                    if 'Năm' in df.columns and 'Kỳ' in df.columns:
                        df['Period'] = df['Năm'].astype(str) + "-Q" + df['Kỳ'].astype(str)
                        df.set_index('Period', inplace=True)
                    elif 'yearReport' in df.columns and 'quarterReport' in df.columns:
                        df['Period'] = df['yearReport'].astype(str) + "-Q" + df['quarterReport'].astype(str)
                        df.set_index('Period', inplace=True)
                    dfs.append(df)
                except: dfs.append(pd.DataFrame())
            self.raw_reports = pd.concat(dfs, axis=1)
            self.raw_reports = self.raw_reports.loc[:, ~self.raw_reports.columns.duplicated()]

            try:
                ratio = self.finance.ratio(period='quarter', lang='vi')
                if not ratio.empty:
                   year = ratio[('Meta', 'Năm')]
                   quarter = ratio[('Meta', 'Kỳ')]
                   ratio['Period'] = year.astype(str) + "-Q" + quarter.astype(str)
                   ratio.set_index('Period', inplace=True)
                   self.ratio_df = ratio
            except Exception as e:
                print(f"Không tải được chỉ số ratio: {e}")

            return not self.raw_reports.empty
        except: return False

    def _get_val(self, keywords):
        if self.raw_reports.empty: return 0.0
        target_col = None
        for k in keywords:
            matches = [col for col in self.raw_reports.columns if k.lower() == col.lower()]
            if not matches: matches = [col for col in self.raw_reports.columns if k.lower() in col.lower()]
            if matches: target_col = matches[0]; break
        if target_col:
            series = pd.to_numeric(self.raw_reports[target_col], errors='coerce').fillna(0)
            if series.abs().max() > 100_000_000_000: return series / 1_000_000_000
            return series
        return pd.Series(0.0, index=self.raw_reports.index)

    def calculate_metrics(self):
        if self.raw_reports.empty: return
        industry = self.profile_info.get('industry', '').lower()

        revenue = self._get_val(['Doanh thu thuần', 'Tổng thu nhập hoạt động'])
        net_income = self._get_val(['Lợi nhuận sau thuế của Cổ đông công ty mẹ', 'Cổ đông của Công ty mẹ'])
        gross_profit = self._get_val(['Lợi nhuận gộp', 'Thu nhập lãi thuần'])
        cogs = self._get_val(['Giá vốn hàng bán', 'Chi phí lãi và các khoản tương tự'])
        equity = self._get_val(['VỐN CHỦ SỞ HỮU'])
        liabilities = self._get_val(['NỢ PHẢI TRẢ'])
        cur_liab = self._get_val(['Nợ ngắn hạn'])
        cur_asset = self._get_val(['TÀI SẢN NGẮN HẠN'])
        inventory = self._get_val(['Hàng tồn kho'])
        ocf = self._get_val(['Lưu chuyển tiền thuần từ HĐKD'])
        capex = self._get_val(['Tiền chi mua sắm', 'Mua sắm TSCĐ'])
        cost_to_operate = (self._get_val(['Chi phí quản lý DN'])).abs()
        total_venue_activity = self._get_val(['Tổng thu nhập hoạt động'])

        #Cho ngan hang
        customer_loan_money = self._get_val(['Cho vay khách hàng'])
        customer_money = self._get_val(['Tiền gửi của khách hàng'])
        value_paper = self._get_val(['Phát hành giấy tờ có giá'])
        money_in_VN_bank = self._get_val(['Tiền gửi tại ngân hàng nhà nước Việt Nam'])
        money_in_other_bank = self._get_val(['Tiền gửi và cho vay các TCTD'])
        money_from_business = (
            self._get_val(['Chứng khoán kinh doanh']) + 
            self._get_val(['Chứng khoán đầu tư sẵn sàng để bán']) + 
            self._get_val(['Chứng khoán đầu tư giữ đến ngày đáo hạn'])
        )

        metrics = pd.DataFrame()
        def safe_div(a, b): return a / b.replace(0, float('nan'))
        
        revenue = revenue.sort_index()
        net_income = net_income.sort_index()

        if not self.ratio_df.empty:
            try:
                metrics['EPS (VND)'] = self.ratio_df[('Chỉ tiêu định giá', 'EPS (VND)')]
                metrics['P/E (Lần)'] = self.ratio_df[('Chỉ tiêu định giá', 'P/E')]
                metrics['P/B (Lần)'] = self.ratio_df[('Chỉ tiêu định giá', 'P/B')]
            except KeyError:
                pass
        
        metrics['Doanh thu (Tỷ)'] = revenue
        metrics['Lợi nhuận (Tỷ)'] = net_income
        metrics['Tăng trưởng DT (YoY %)'] = revenue.pct_change(periods=4) * 100
        metrics['Tăng trưởng LN (YoY %)'] = net_income.pct_change(periods=4) * 100
        metrics['Biên LN Gộp (%)'] = safe_div(gross_profit, revenue) * 100
        metrics['Biên LN Ròng (%)'] = safe_div(net_income, revenue) * 100
        metrics['ROE (Quý) (%)'] = safe_div(net_income, equity) * 100 
        metrics['Nợ/Vốn chủ (Lần)'] = safe_div(liabilities, equity)
        metrics['CIR (Lần)'] = safe_div(cost_to_operate,total_venue_activity)
        
        if 'ngân hàng' in industry or 'bank' in industry:
            metrics['LDR (%)'] = safe_div(customer_loan_money, (customer_money + value_paper)) * 100
            metrics['Vòng quay kho'] = 0
            metrics['FCF (Tỷ)'] = 0
            metrics['Thanh toán hiện hành (Lần)'] = 0
            total_earning_assets = customer_loan_money + money_in_VN_bank + money_in_other_bank + money_from_business
            tai_san_sinh_loi_bq = total_earning_assets.rolling(window=2).mean()
            metrics['NIM (%)'] = safe_div(gross_profit*4,tai_san_sinh_loi_bq)*100
            metrics['Biên LN Gộp (%)'] = 0
        else: 
            metrics['Biên LN Gộp (%)'] = safe_div(gross_profit, revenue) * 100
            metrics['LDR (%)'] = 0
            metrics['Vòng quay kho'] = safe_div(cogs, inventory)
            metrics['FCF (Tỷ)'] = ocf + capex
            metrics['Thanh toán hiện hành (Lần)'] = safe_div(cur_asset, cur_liab)
            metrics['NIM (%)'] = 0

        self.final_metrics = metrics.round(2).sort_index(ascending=True)
        return self.final_metrics

    # --- 3. TRỰC QUAN HÓA ---
    def visualize_stock_price(self):
        if self.price_history.empty: return
        df = self.price_history.sort_index()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        
        last_price = df['close'].iloc[-1]
        last_sma = df['SMA50'].iloc[-1]
        trend = "TĂNG" if last_price > last_sma else "GIẢM"
        
        text_str = f'Giá hiện tại: {last_price:,.0f} đ\nXu hướng (vs SMA50): {trend}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')

        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Giá đóng cửa (VND)', color='#1f77b4', linewidth=1.5)
        plt.plot(df.index, df['SMA50'], label='SMA 50 (Xu hướng trung hạn)', color='orange', linestyle='--', linewidth=1.5)
        
        plt.text(0.02, 0.95, text_str, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontweight='bold', color='black')

        plt.title(f'BIỂU ĐỒ GIÁ CỔ PHIẾU {self.symbol} (1 NĂM)', fontsize=14, fontweight='bold')
        plt.ylabel('Giá cổ phiếu (VND)')
        plt.legend(loc='lower left', title='CHÚ GIẢI:')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def visualize_financials(self, df):
        try:
            plot_data = df.tail(8)
            
            df_display = plot_data.T.reset_index()
            df_display.rename(columns={'index': 'CHỈ TIÊU'}, inplace=True)
            
            print(f"\n>>> BẢNG CHI TIẾT CÁC CHỈ SỐ (8 Quý gần nhất):")
            print("-" * 100)
            print(df_display.to_string(index=False)) 
            print("-" * 100)
            
            cols_growth = ['Tăng trưởng DT (YoY %)', 'Tăng trưởng LN (YoY %)']
            if all(c in plot_data.columns for c in cols_growth):
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_data[cols_growth].plot(kind='bar', ax=ax, width=0.7)
                
                ax.set_title(f'{self.symbol} - TỐC ĐỘ TĂNG TRƯỞNG (YOY)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Phần trăm (%)')
                ax.set_xlabel('Quý báo cáo')
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.legend(['Cột xanh: Tăng trưởng Doanh thu', 'Cột cam: Tăng trưởng Lợi nhuận'], 
                          title='CHÚ GIẢI:', loc='best')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()

            cols_val = ['P/E (Lần)', 'P/B (Lần)']
            valid_val = [c for c in cols_val if c in plot_data.columns]
            if valid_val:
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_data[valid_val].plot(kind='line', marker='o', linewidth=2, ax=ax)
                ax.set_title(f'{self.symbol} - ĐỊNH GIÁ (P/E & P/B)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Lần'); ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(valid_val, loc='best')
                plt.tight_layout(); plt.show()

            cols_margin = ['Biên LN Gộp (%)', 'Biên LN Ròng (%)', 'ROE (Quý) (%)']
            valid_cols = [c for c in cols_margin if c in plot_data.columns]
            
            if valid_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                plot_data[valid_cols].plot(kind='line', marker='o', linewidth=2, ax=ax)
                ax.set_title(f'{self.symbol} - HIỆU QUẢ SINH LỜI & ROE', fontsize=14, fontweight='bold')
                ax.set_ylabel('Phần trăm (%)')
                ax.set_xlabel('Quý báo cáo')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(valid_cols, title='CHÚ GIẢI:', loc='best')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Lỗi hiển thị biểu đồ: {e}")

    def visualize_ownership(self):
        df_sh = self.profile_info.get('shareholders')
        if df_sh is None or df_sh.empty: return
        try:
            df_plot = df_sh.copy()
            df_plot['share_own_percent'] = df_plot['share_own_percent'].fillna(0)
            if df_plot['share_own_percent'].max() <= 1.0: df_plot['share_own_percent'] *= 100
            
            df_plot = df_plot.sort_values('share_own_percent', ascending=False)
            top_5 = df_plot.head(5)
            others = max(0, 100 - top_5['share_own_percent'].sum())
            
            labels = list(top_5['share_holder'])
            sizes = list(top_5['share_own_percent'])
            if others > 0.1: labels.append('Cổ đông khác'); sizes.append(others)

            fig, ax = plt.subplots(figsize=(9, 5))
            wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%', 
                                              startangle=90, colors=sns.color_palette('pastel'), pctdistance=0.85)
            ax.legend(wedges, labels, title="Danh sách Cổ đông", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            fig.gca().add_artist(plt.Circle((0,0),0.70,fc='white'))
            plt.title(f'CƠ CẤU SỞ HỮU - {self.symbol}', fontweight='bold')
            plt.tight_layout()
            plt.show()
        except: pass

    def display_text_table(self, data, date_col, title_col, link_col, table_title):
        """Hàm in bảng Text cho file .py (thay vì HTML)"""
        if not data: return
        print(f"\n>>> {table_title}:")
        print("-" * 80)
        for row in data:
            date = row.get(date_col, 'N/A')
            title = row.get(title_col, 'N/A')
            link = row.get(link_col, '')
            print(f"[{date}] {title}")
            if link:
                print(f"   Link: {link}")
        print("-" * 80)

    def display_glossary(self):
        print("\n" + "="*50)
        print("📖 BẢNG GIẢI THÍCH THUẬT NGỮ (GLOSSARY)")
        print("="*50)
        glossary = [
            # --- NHÓM 1: CƠ BẢN (GENERAL) ---
            ["Doanh thu (TOI)", "Tổng thu nhập hoạt động (Lãi thuần + Dịch vụ + Khác)."],
            ["Lợi nhuận ròng", "Lãi sau thuế dành cho cổ đông công ty mẹ."],
            ["YoY (%)", "Tốc độ tăng trưởng so với cùng kỳ năm trước."],
            ["ROE (%)", "Hiệu quả sinh lời trên vốn chủ sở hữu (Nhà đầu tư bỏ 100đ lãi được bao nhiêu?)."],
            ["D/E (Lần)", "Đòn bẩy tài chính: Tỷ lệ Nợ vay / Vốn chủ sở hữu."],

            # --- NHÓM 2: CHỈ SỐ SỐNG CÒN CỦA NGÂN HÀNG (BANKING KPIs) ---
            ["NIM (%)", "Biên lãi ròng: Chênh lệch thực tế giữa lãi cho vay và lãi huy động."],
            ["LDR (%)", "Dư nợ/Huy động: Mức độ tối ưu vốn (Cao quá = Rủi ro thanh khoản)."],
            ["CIR (Lần)", "Chi phí/Thu nhập: Tốn bao nhiêu đồng phí để kiếm 1 đồng doanh thu (Thấp là tốt)."],

            # --- NHÓM 3: ĐỊNH GIÁ (VALUATION - Nên có để biết đắt/rẻ) ---
            ["EPS (VND)", "Lãi cơ bản trên mỗi cổ phiếu."],
            ["P/E (Lần)", "Giá/Lợi nhuận: Số năm hòa vốn với mức lợi nhuận hiện tại."],
            ["P/B (Lần)", "Giá/Sổ sách: Giá cổ phiếu cao gấp mấy lần giá trị tài sản ròng."],
        ]
        for term, desc in glossary:
            print(f"• {term:<25} : {desc}")

    def display_report(self):
        info = self.profile_info
        print("\n" + "="*80)
        print(f"BÁO CÁO PHÂN TÍCH: {self.symbol} - {info.get('industry', 'N/A')}")
        print("-" * 80)
        print(f"Giá: {info.get('price'):,} ({info.get('pct_change')*100:.2f}%) | Vốn hóa: {info.get('market_cap', 0)/1e9:,.0f} Tỷ")
        
        if info['officers']:
            print(f"\nBAN LÃNH ĐẠO:")
            for p in info['officers'][:5]: print(f" - {p.get('officer_name')} ({p.get('officer_position')})")
        
        self.display_text_table(info['news'], 'date_str', 'news_title', 'news_source_link', f"TIN TỨC MỚI NHẤT")
        self.display_text_table(info['events'], 'date_str', 'display_name', 'event_link', f"SỰ KIỆN DOANH NGHIỆP")

        print("\n" + "="*80)
        print("\n>>> XU HƯỚNG GIÁ & KỸ THUẬT:")
        self.visualize_stock_price()

        if self.final_metrics is not None:
            print("\n>>> TRỰC QUAN HÓA CHỈ SỐ TÀI CHÍNH:")
            self.visualize_financials(self.final_metrics.loc[:, (self.final_metrics != 0).any(axis=0)])
        
        print("\n>>> CƠ CẤU CỔ ĐÔNG:")
        self.visualize_ownership()
        
        self.display_glossary()

if __name__ == "__main__":
    try:
        symbol = input("Nhập mã cổ phiếu (VD: ACB, HPG): ")
        app = BusinessAnalyzer(symbol)
        app.get_company_info()
        app.get_historical_price()
        if app.get_financial_data():
            app.calculate_metrics()
            app.display_report()
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")