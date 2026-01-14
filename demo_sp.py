import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime, timedelta
from vnstock import Finance, Company, Quote

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Cấu hình hiển thị pandas
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
            'news': [], 'events': [], 'market_cap': 0,
            'industry': '', 'industry2': '', 'exchange': '', 'price': 0, 'pct_change': 0
        } 

    def get_company_info(self):
        print(f"--- Đang tải thông tin {self.symbol} (Nguồn: VCI) ---")
        try:
            # 1. Thông tin chung
            overview = self.company.overview()
            if not overview.empty:
                item = overview.iloc[0]
                self.profile_info.update({
                    'industry': item.get('icb_name2', ''),
                    'industry2': item.get('icb_name4', ''),
                    'exchange': item.get('exchange', 'VN')
                })
            
            # 2. Giá cả
            stats = self.company.trading_stats()
            if not stats.empty:
                item = stats.iloc[0]
                self.profile_info.update({
                    'price': item.get('close_price', 0),
                    'pct_change': item.get('price_change_pct', 0)
                })
            
            # 3. Thông tin phụ
            self._fetch_sub_info()
            
            # 4. MARKET CAP (Đã sửa lại logic truy cập MultiIndex chuẩn xác nhất)
            try:
                # Ưu tiên 1: Lấy từ trading_stats (thường chuẩn xác nhất)
                if not stats.empty and 'market_cap' in stats.columns:
                    val = float(stats.iloc[0]['market_cap'])
                    if val > 0:
                        # Logic kiểm tra đơn vị: Nếu < 1 triệu tỷ -> khả năng là đơn vị Tỷ -> nhân 1 tỷ
                        if val < 1_000_000_000: 
                            self.profile_info['market_cap'] = val * 1_000_000_000
                        else: # Nếu số quá lớn -> đã là đơn vị Đồng
                            self.profile_info['market_cap'] = val

                # Ưu tiên 2: Nếu chưa có, lấy từ Ratio (như code cũ nhưng thêm kiểm tra)
                if self.profile_info['market_cap'] == 0:
                    if self.ratio_df.empty:
                        self.ratio_df = self.finance.ratio(period='quarter', lang='vi')
                    
                    if not self.ratio_df.empty:
                        # Sort để lấy kỳ mới nhất
                        if ('Meta', 'Năm') in self.ratio_df.columns:
                            y = self.ratio_df[('Meta', 'Năm')]
                            q = self.ratio_df[('Meta', 'Kỳ')]
                            self.ratio_df['_sort_val'] = y * 10 + q
                            self.ratio_df = self.ratio_df.sort_values('_sort_val')

                        cap_col = self.ratio_df.get(('Chỉ tiêu định giá', 'Vốn hóa (Tỷ đồng)'))
                        if cap_col is not None and not cap_col.empty:
                            raw_val = float(cap_col.iloc[-1]) # Ép kiểu float để tránh tràn số
                            
                            # LOGIC QUAN TRỌNG: Kiểm tra độ lớn của số
                            # Nếu raw_val > 100 tỷ -> Đã là đơn vị Đồng -> Giữ nguyên
                            # Nếu raw_val < 100 tỷ (VD: 264,000) -> Là đơn vị Tỷ -> Nhân 1 tỷ
                            if raw_val > 100_000_000_000:
                                self.profile_info['market_cap'] = raw_val
                            else:
                                self.profile_info['market_cap'] = raw_val * 1_000_000_000
                                
            except Exception as e:
                print(f"Lỗi tính vốn hóa: {e}")

            return True
        except Exception as e: 
            print(f"Lỗi tải thông tin công ty: {e}")
            return False

    def _fetch_sub_info(self):
        try: self.profile_info['officers'] = self.company.officers().head(10).to_dict('records')
        except: pass
        try: self.profile_info['shareholders'] = self.company.shareholders()
        except: pass
        try: 
            df_news = self.company.news()
            if not df_news.empty:
                latest = df_news.head(5).copy()
                latest['date_str'] = pd.to_datetime(latest.get('public_date'), unit='ms', errors='coerce').dt.strftime('%d/%m/%Y').fillna('N/A')
                if 'news_source_link' not in latest.columns: latest['news_source_link'] = ''
                self.profile_info['news'] = latest.to_dict('records')
        except: pass
        try:
            df_evt = self.company.events()
            if not df_evt.empty:
                evt = df_evt.copy()
                date_col = next((c for c in ['exright_date', 'public_date', 'notify_date'] if c in evt.columns), None)
                if date_col:
                    evt['sort_date'] = pd.to_datetime(evt[date_col], errors='coerce')
                    evt = evt.sort_values('sort_date', ascending=False).head(5)
                    evt['date_str'] = evt['sort_date'].dt.strftime('%d/%m/%Y').fillna('N/A')
                else: evt['date_str'] = 'N/A'
                evt['event_link'] = evt.get('source_url', '')
                if 'event_title' in evt.columns: evt['display_name'] = evt['event_title']
                else: evt['display_name'] = 'Sự kiện doanh nghiệp'
                self.profile_info['events'] = evt.to_dict('records')
        except: pass

    def get_historical_price(self):
        try:
            end = datetime.now(); start = end - timedelta(days=365)
            df = self.quote.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']); df.set_index('time', inplace=True)
                self.price_history = df; return True
        except: return False
        return False

    def get_financial_data(self):
        print(f"--- Đang tải báo cáo tài chính (Theo Quý)... ---")
        try:
            dfs = []
            for func in [self.finance.income_statement, self.finance.balance_sheet, self.finance.cash_flow]:
                try:
                    df = func(period='quarter', lang='vi', dropna=True)
                    # Tạo Index chuẩn "Năm-Quý" để tránh lỗi NaN
                    if 'Năm' in df.columns and 'Kỳ' in df.columns:
                        df['Period'] = df['Năm'].astype(str) + "-Q" + df['Kỳ'].astype(str)
                    elif 'yearReport' in df.columns and 'quarterReport' in df.columns:
                        df['Period'] = df['yearReport'].astype(str) + "-Q" + df['quarterReport'].astype(str)
                    
                    if 'Period' in df.columns:
                        df.set_index('Period', inplace=True)
                        df = df[~df.index.duplicated(keep='first')]
                    dfs.append(df)
                except: dfs.append(pd.DataFrame())
            
            self.raw_reports = pd.concat(dfs, axis=1)
            self.raw_reports = self.raw_reports.loc[:, ~self.raw_reports.columns.duplicated()]
            
            # Đồng bộ index cho Ratio DF để tính metrics (P/E, P/B...)
            if not self.ratio_df.empty and 'Period' not in self.ratio_df.columns:
                try:
                    if ('Meta', 'Năm') in self.ratio_df.columns:
                        y = self.ratio_df[('Meta', 'Năm')]
                        q = self.ratio_df[('Meta', 'Kỳ')]
                        self.ratio_df['Period'] = y.astype(str) + "-Q" + q.astype(str)
                        self.ratio_df.set_index('Period', inplace=True)
                except: pass

            return not self.raw_reports.empty
        except: return False

    def _get_val(self, keywords):
        if self.raw_reports.empty: return 0.0
        if isinstance(keywords, str): keywords = [keywords]
        for k in keywords:
            matches = [c for c in self.raw_reports.columns if k.lower() == c.lower()]
            if not matches: matches = [c for c in self.raw_reports.columns if k.lower() in c.lower()]
            if matches:
                s = pd.to_numeric(self.raw_reports[matches[0]], errors='coerce').fillna(0)
                if s.abs().max() > 100_000_000_000: return s / 1_000_000_000
                return s
        return pd.Series(0.0, index=self.raw_reports.index)

    def calculate_metrics(self):
        if self.raw_reports.empty: return

        # Lấy dữ liệu cơ sở
        revenue = self._get_val(['Doanh thu thuần', 'Tổng thu nhập hoạt động'])
        net_income = self._get_val(['Lợi nhuận sau thuế của Cổ đông công ty mẹ', 'Cổ đông của Công ty mẹ', 'Lợi nhuận sau thuế'])
        gross_profit = self._get_val(['Lợi nhuận gộp', 'Thu nhập lãi thuần']) # Bank dùng Thu nhập lãi thuần
        cogs = self._get_val(['Giá vốn hàng bán', 'Chi phí lãi và các khoản tương tự'])
        equity = self._get_val(['VỐN CHỦ SỞ HỮU'])
        
        cur_liab = self._get_val(['Nợ ngắn hạn (đồng)', 'Nợ ngắn hạn'])
        cur_asset = self._get_val(['TÀI SẢN NGẮN HẠN', 'Tài sản ngắn hạn (đồng)'])
        inventory = self._get_val(['Hàng tồn kho', 'Hàng tồn kho (đồng)'])
        total_assets = self._get_val(['TỔNG CỘNG TÀI SẢN (đồng)', 'TỔNG CỘNG TÀI SẢN'])
        
        ocf = self._get_val(['Lưu chuyển tiền thuần từ HĐKD'])
        capex = self._get_val(['Tiền chi mua sắm', 'Mua sắm TSCĐ', 'Tiền chi mua sắm, xây dựng TSCĐ'])
        
        cost_op = self._get_val(['Chi phí quản lý DN', 'Chi phí quản lý doanh nghiệp']).abs()
        
        # Tạo bảng kết quả
        metrics = pd.DataFrame()
        def safe_div(a, b): return a / b.replace(0, float('nan'))
        
        revenue = revenue.sort_index()
        net_income = net_income.sort_index()

        # Thêm chỉ số từ API Ratio
        if not self.ratio_df.empty:
            try:
                # Dùng .get để tránh lỗi KeyError nếu cột không tồn tại
                metrics['EPS (VND)'] = self.ratio_df.get(('Chỉ tiêu định giá', 'EPS (VND)'), 0)
                metrics['P/E (Lần)'] = self.ratio_df.get(('Chỉ tiêu định giá', 'P/E'), 0)
                metrics['P/B (Lần)'] = self.ratio_df.get(('Chỉ tiêu định giá', 'P/B'), 0)
                metrics['ROE (Quý) (%)'] = self.ratio_df.get(('Chỉ tiêu khả năng sinh lợi', 'ROE (%)'), 0)
            except: pass

        metrics['Doanh thu (Tỷ)'] = revenue
        metrics['Lợi nhuận (Tỷ)'] = net_income
        metrics['Tăng trưởng DT (YoY %)'] = revenue.pct_change(periods=4) * 100
        metrics['Tăng trưởng LN (YoY %)'] = net_income.pct_change(periods=4) * 100
        metrics['Biên LN Ròng (%)'] = safe_div(net_income, revenue) * 100
        
        industry = self.profile_info.get('industry', '').lower()
        industry2 = self.profile_info.get('industry2', '').lower()
        
        # --- LOGIC NGÀNH ---
        if 'ngân hàng' in industry or 'bank' in industry:
            loans = self._get_val(['Cho vay khách hàng'])
            deposits = self._get_val(['Tiền gửi của khách hàng']) + self._get_val(['Phát hành giấy tờ có giá'])
            earning_assets = (loans + self._get_val(['Chứng khoán đầu tư']) + self._get_val(['Tiền gửi tại NHNN']))
            prov = self._get_val(['Chi phí dự phòng rủi ro tín dụng'])
            
            metrics['LDR (%)'] = safe_div(loans, deposits) * 100
            metrics['NIM (%)'] = safe_div(gross_profit * 4, earning_assets.rolling(2).mean()) * 100
            metrics['CIR (%)'] = safe_div(cost_op, revenue) * 100
            metrics['Chi phí dự phòng/Dư nợ (%)'] = safe_div(prov, loans).abs() * 100

        elif 'dịch vụ tài chính' in industry: # Chứng khoán
            fvtpl = self._get_val(['Giá trị thuần đầu tư ngắn hạn (đồng)', 'Tài sản tài chính ghi nhận thông qua lãi/lỗ'])
            margin = self._get_val(['Các khoản phải thu ngắn hạn (đồng)', 'Các khoản cho vay'])
            metrics['FVTPL/Tổng TS (%)'] = safe_div(fvtpl, total_assets) * 100
            metrics['Margin/Vốn chủ (Lần)'] = safe_div(margin, equity)

        elif 'bất động sản' in industry:
            prepay = self._get_val(['Người mua trả tiền trước ngắn hạn'])
            metrics['Người mua trả trước (Tỷ)'] = prepay
            metrics['Hàng tồn kho (Tỷ)'] = inventory
            metrics['Tỷ lệ Trả trước/Tồn kho (%)'] = safe_div(prepay, inventory) * 100
            metrics['FCF (Tỷ)'] = ocf - capex

        elif 'bảo hiểm' in industry:
            inv_short = self._get_val(['Giá trị thuần đầu tư ngắn hạn (đồng)'])
            inv_long = self._get_val(['Đầu tư dài hạn (đồng)'])
            cash = self._get_val(['Tiền và tương đương tiền (đồng)'])
            total_inv = inv_short + inv_long + cash

            # 2. HIỆU QUẢ ĐẦU TƯ
            # Dựa trên Income Statement: Thu nhập TC - Chi phí TC
            fin_income = self._get_val(['Thu nhập tài chính'])
            fin_cost = self._get_val(['Chi phí tài chính'])
            net_inv_income = fin_income - fin_cost

            # 3. CHI PHÍ HOẠT ĐỘNG
            sell_exp = self._get_val(['Chi phí bán hàng']).abs()
            admin_exp = self._get_val(['Chi phí quản lý DN']).abs()

            # --- TÍNH TOÁN CHỈ SỐ ---
            
            # Tỷ trọng tài sản đầu tư / Tổng tài sản
            # Cho biết bao nhiêu % tài sản đang sinh lời từ hoạt động tài chính
            metrics['Đầu tư/Tổng TS (%)'] = safe_div(total_inv, total_assets) * 100

            # Lợi suất đầu tư (ROI)
            # Hiệu quả của mảng đầu tư tài chính
            metrics['Lợi suất đầu tư (%)'] = safe_div(net_inv_income, total_inv) * 100

            # Biên Lợi nhuận gộp (Đại diện cho hiệu quả HĐ Bảo hiểm gốc)
            # Vì không có cột 'Chi bồi thường', ta dùng Lãi gộp làm proxy.
            # Lãi gộp cao đồng nghĩa Doanh thu phí bù đắp tốt cho Bồi thường & Dự phòng.
            metrics['Biên lãi gộp BH (%)'] = safe_div(gross_profit, revenue) * 100

            # Tỷ lệ chi phí hoạt động (Expense Ratio Proxy)
            # (Chi phí bán hàng + QLDN) / Doanh thu thuần
            metrics['Tỷ lệ chi phí HĐ (%)'] = safe_div(sell_exp + admin_exp, revenue) * 100
            
            # Đòn bẩy tài chính (Tổng TS / Vốn chủ)
            # Bảo hiểm thường có đòn bẩy cao do chiếm dụng vốn (Float)
            metrics['Đòn bẩy tài chính (Lần)'] = safe_div(total_assets, equity)

            # Khả năng thanh toán (Dựa trên cột Ratio bạn cung cấp)
            if not self.ratio_df.empty:
                 metrics['Thanh toán hiện hành'] = self.ratio_df.get(('Chỉ tiêu thanh khoản', 'Chỉ số thanh toán hiện thời'), 0)

        elif 'sản xuất và khai thác dầu khí' in industry2 or 'điện, nước & xăng dầu khí đốt' in industry:
            fixed_assets = self._get_val(['Tài sản cố định (đồng)', 'Tài sản cố định'])
            
            depreciation = self._get_val(['Khấu hao TSCĐ'])
            
            op_profit = self._get_val(['Lãi/Lỗ từ hoạt động kinh doanh', 'Lợi nhuận thuần từ hoạt động kinh doanh'])
            
            total_debt = self._get_val(['Vay và nợ thuê tài chính ngắn hạn (đồng)']) + \
                         self._get_val(['Vay và nợ thuê tài chính dài hạn (đồng)'])
            cash = self._get_val(['Tiền và tương đương tiền (đồng)'])

            ebitda = op_profit + depreciation
            metrics['EBITDA (Tỷ)'] = ebitda
            metrics['Biên EBITDA (%)'] = safe_div(ebitda, revenue) * 100

            net_debt = total_debt - cash
            metrics['Vay ròng/EBITDA (Lần)'] = safe_div(net_debt, ebitda)

            metrics['Vòng quay TSCĐ'] = safe_div(revenue, fixed_assets)

            metrics['Tỷ trọng TSCĐ (%)'] = safe_div(fixed_assets, total_assets) * 100

            metrics['FCF (Tỷ)'] = ocf + capex 
            
            div_cash_paid = self._get_val(['Cổ tức đã trả']).abs()
            metrics['Cổ tức tiền mặt đã trả (Tỷ)'] = div_cash_paid

        else: # Sản xuất / Thương mại
            metrics['Biên LN Gộp (%)'] = safe_div(gross_profit, revenue) * 100
            metrics['Vòng quay kho'] = safe_div(cogs, inventory)
            metrics['Thanh toán hiện hành'] = safe_div(cur_asset, cur_liab)
            metrics['FCF (Tỷ)'] = ocf + capex 
            
            # Vay nợ
            total_debt = self._get_val(['Vay và nợ thuê tài chính ngắn hạn']) + self._get_val(['Vay và nợ thuê tài chính dài hạn'])
            cash = self._get_val(['Tiền và tương đương tiền'])
            metrics['Vay ròng/Vốn chủ (Lần)'] = safe_div(total_debt - cash, equity)

        self.final_metrics = metrics.round(2).sort_index()
        return self.final_metrics

    # --- DISPLAY ---
    def visualize_stock_price(self):
        if self.price_history.empty: return
        df = self.price_history.sort_index()
        df['SMA50'] = df['close'].rolling(50).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Giá', color='#1f77b4')
        plt.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='orange')
        plt.title(f'BIỂU ĐỒ GIÁ {self.symbol}'); plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout(); plt.show()

    def visualize_financials(self, df):
        try:
            plot_data = df.tail(8)
            print(f"\n>>> BẢNG CHI TIẾT CÁC CHỈ SỐ (8 Quý gần nhất):")
            print("-" * 120)
            print(plot_data.T.reset_index().rename(columns={'index': 'CHỈ TIÊU'}).to_string(index=False))
            print("-" * 120)
            
            cols_growth = ['Tăng trưởng DT (%)', 'Tăng trưởng LN (%)']
            if all(c in plot_data.columns for c in cols_growth):
                plot_data[cols_growth].plot(kind='bar', figsize=(12, 6), title=f'{self.symbol} - TĂNG TRƯỞNG YoY')
                plt.axhline(0, color='black', lw=1); plt.show()
        except: pass

    def visualize_ownership(self):
        df_sh = self.profile_info.get('shareholders')
        if df_sh is None or df_sh.empty: return
        try:
            df_plot = df_sh.copy().sort_values('share_own_percent', ascending=False)
            df_plot['share_own_percent'] = df_plot['share_own_percent'].fillna(0)
            if df_plot['share_own_percent'].max() <= 1.0: df_plot['share_own_percent'] *= 100
            
            top_5 = df_plot.head(5)
            others = max(0, 100 - top_5['share_own_percent'].sum())
            labels = list(top_5['share_holder']) + (['Khác'] if others > 0 else [])
            sizes = list(top_5['share_own_percent']) + ([others] if others > 0 else [])

            fig, ax = plt.subplots(figsize=(14, 7))
            wedges, _, _ = ax.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'), pctdistance=0.85)
            ax.legend(wedges, labels, title="Cổ đông", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.title(f'CƠ CẤU SỞ HỮU - {self.symbol}', fontweight='bold'); plt.tight_layout(); plt.show()
        except: pass

    def display_text_table(self, data, date_col, title_col, link_col, table_title):
        if not data: return
        print(f"\n>>> {table_title}:")
        print("-" * 100)
        for row in data:
            d = str(row.get(date_col, 'N/A'))
            t = str(row.get(title_col, 'N/A'))
            l = str(row.get(link_col, ''))
            print(f"{d:<12} | {t}")
            if l and l != 'nan': print(f"{'':<12}   Link: {l}")
        print("-" * 100)

    def display_report(self):
        info = self.profile_info
        print("\n" + "="*80)
        print(f"BÁO CÁO PHÂN TÍCH: {self.symbol} - {info.get('industry', 'N/A')}")
        print("-" * 80)
        
        # Xử lý hiển thị Vốn hóa
        m_cap = info.get('market_cap', 0)
        if m_cap > 0:
            cap_str = f"{m_cap/1_000_000_000:,.0f} Tỷ"
        else:
            cap_str = "N/A"

        print(f"Giá: {info.get('price', 0):,} ({info.get('pct_change', 0)*100:.2f}%) | Vốn hóa: {cap_str}")
        
        if info['officers']:
            print(f"\n>>> BAN LÃNH ĐẠO:")
            for p in info['officers'][:5]: print(f" - {p.get('officer_name')} ({p.get('officer_position', 'N/A')})")
        
        self.display_text_table(info['news'], 'date_str', 'news_title', 'news_source_link', "TIN TỨC")
        self.display_text_table(info['events'], 'date_str', 'display_name', 'event_link', "SỰ KIỆN DOANH NGHIỆP")

        print("\n>>> XU HƯỚNG GIÁ:"); self.visualize_stock_price()
        if self.final_metrics is not None:
            valid_cols = self.final_metrics.columns[(self.final_metrics != 0).any() & (self.final_metrics.notna().any())]
            self.visualize_financials(self.final_metrics[valid_cols])
        print("\n>>> CƠ CẤU CỔ ĐÔNG:"); self.visualize_ownership()

if __name__ == "__main__":
    try:
        symbol = input("Nhập mã cổ phiếu (VD: ACB, HPG): ")
        if symbol:
            app = BusinessAnalyzer(symbol)
            app.get_company_info()
            app.get_historical_price()
            if app.get_financial_data():
                app.calculate_metrics()
                app.display_report()
    except Exception as e: print(f"Lỗi: {e}")