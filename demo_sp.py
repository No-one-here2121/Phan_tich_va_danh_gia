import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from datetime import datetime, timedelta
from vnstock import Finance, Company, Quote, Listing

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels không được cài đặt. Chức năng seasonal decomposition bị tắt.")

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000) 
pd.options.display.float_format = '{:,.2f}'.format

class SeasonalNaNHandler:
    """Xử lý giá trị NaN bằng Seasonal Decomposition và các phương pháp dự phòng"""
    
    def __init__(self, period=4):
        """
        Args:
            period: Chu kỳ mùa vụ (4 cho dữ liệu theo quý, 12 cho theo tháng)
        """
        self.period = period
        self.min_observations = period * 2  # Tối thiểu 2 chu kỳ để phân tách
    
    def fill_nan_seasonal(self, series, method='additive'):
        """
        Điền NaN bằng seasonal decomposition
        
        Args:
            series: pandas Series chứa dữ liệu cần xử lý
            method: 'additive' hoặc 'multiplicative'
        
        Returns:
            pandas Series đã được điền NaN
        """
        if not HAS_STATSMODELS:
            return self._fallback_fill(series)
        
        # Loại bỏ NaN để kiểm tra độ dài
        non_nan_count = series.notna().sum()
        
        # Nếu không đủ dữ liệu, dùng phương pháp dự phòng
        if non_nan_count < self.min_observations:
            return self._fallback_fill(series)
        
        # Nếu không có NaN, trả về nguyên bản
        if series.isna().sum() == 0:
            return series
        
        try:
            # Tạo bản sao để xử lý
            filled_series = series.copy()
            
            # Bước 1: Điền tạm thời bằng interpolation để có đủ dữ liệu cho decomposition
            temp_filled = filled_series.interpolate(method='linear', limit_direction='both')
            
            # Nếu vẫn còn NaN (ở đầu/cuối), dùng forward/backward fill
            temp_filled = temp_filled.ffill().bfill()
            
            # Bước 2: Áp dụng seasonal decomposition
            decomposition = seasonal_decompose(
                temp_filled, 
                model=method, 
                period=self.period,
                extrapolate_trend='freq'
            )
            
            # Bước 3: Chỉ điền vào các vị trí NaN ban đầu
            # Sử dụng trend + seasonal để dự đoán
            predicted = decomposition.trend + decomposition.seasonal
            
            # Điền vào các vị trí NaN
            nan_mask = series.isna()
            filled_series[nan_mask] = predicted[nan_mask]
            
            return filled_series
            
        except Exception as e:
            # Nếu seasonal decomposition thất bại, dùng phương pháp dự phòng
            print(f"Cảnh báo: Seasonal decomposition thất bại ({str(e)}), sử dụng phương pháp dự phòng")
            return self._fallback_fill(series)
    
    def _fallback_fill(self, series):
        """
        Phương pháp dự phòng: kết hợp nhiều kỹ thuật
        1. Interpolation (nội suy tuyến tính)
        2. Forward fill
        3. Backward fill
        4. Mean (nếu vẫn còn NaN)
        """
        filled = series.copy()
        
        # Bước 1: Interpolation tuyến tính
        filled = filled.interpolate(method='linear', limit_direction='both')
        
        # Bước 2: Forward fill cho các giá trị đầu
        filled = filled.ffill()
        
        # Bước 3: Backward fill cho các giá trị cuối
        filled = filled.bfill()
        
        # Bước 4: Nếu vẫn còn NaN, dùng mean
        if filled.isna().any():
            filled = filled.fillna(series.mean())
        
        return filled
    
    def fill_dataframe(self, df, columns=None, method='additive'):
        """
        Điền NaN cho toàn bộ DataFrame
        
        Args:
            df: pandas DataFrame
            columns: List các cột cần xử lý (None = tất cả cột số)
            method: 'additive' hoặc 'multiplicative'
        
        Returns:
            pandas DataFrame đã được điền NaN
        """
        result = df.copy()
        
        if columns is None:
            # Lấy tất cả cột số
            columns = result.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in result.columns:
                result[col] = self.fill_nan_seasonal(result[col], method=method)
        
        return result

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
            'industry': '', 'industry2': '', 'exchange': '', 'price': 0, 'pct_change': 0,
            # Thông tin chi tiết công ty
            'organ_name': '', 'short_name': '', 'website': '', 'established_year': '', 'no_employees': 0,
            'no_shareholders': 0, 'foreign_percent': 0, 'outstanding_share': 0, 'issue_share': 0,
            'charter_capital': 0
        } 

    def get_company_info(self):
        print(f"--- Đang tải thông tin {self.symbol} (Nguồn: VCI) ---")
        try:
            # Lấy tên công ty đầy đủ từ Listing
            try:
                listing = Listing()
                all_symbols = listing.all_symbols()
                company_row = all_symbols[all_symbols['symbol'] == self.symbol]
                if not company_row.empty:
                    self.profile_info['organ_name'] = company_row.iloc[0].get('organ_name', '')
            except:
                pass
            
            # 1. Thông tin chung
            overview = self.company.overview()
            if not overview.empty:
                item = overview.iloc[0]
                self.profile_info.update({
                    'industry': item.get('icb_name2', ''),
                    'industry2': item.get('icb_name4', ''),
                    'exchange': item.get('exchange', 'VN'),
                    # Thông tin bổ sung từ overview
                    'short_name': item.get('shortName', item.get('short_name', '')),
                    'website': item.get('website', ''),
                    'established_year': item.get('establishedYear', item.get('established_year', '')),
                    'no_employees': item.get('noEmployees', item.get('no_employees', 0)),
                    'no_shareholders': item.get('noShareholders', item.get('no_shareholders', 0)),
                    'foreign_percent': item.get('foreignPercent', item.get('foreign_percent', 0)),
                    'outstanding_share': item.get('outstandingShare', item.get('outstanding_share', 0)),
                    'issue_share': item.get('issueShare', item.get('issue_share', 0)),
                    'charter_capital': item.get('charterCapital', item.get('charter_capital', 0))
                })
            
            # Giá cả
            stats = self.company.trading_stats()
            if not stats.empty:
                item = stats.iloc[0]
                self.profile_info.update({
                    'price': item.get('close_price', 0),
                    'pct_change': item.get('price_change_pct', 0)
                })
            
            # Thông tin phụ
            self._fetch_sub_info()
            
            try:
                if not stats.empty and 'market_cap' in stats.columns:
                    val = float(stats.iloc[0]['market_cap'])
                    if val > 0:
                        # Logic kiểm tra đơn vị: Nếu < 1 triệu tỷ -> khả năng là đơn vị Tỷ -> nhân 1 tỷ
                        if val < 1_000_000_000: 
                            self.profile_info['market_cap'] = val * 1_000_000_000
                        else: # Nếu số quá lớn -> đã là đơn vị Đồng
                            self.profile_info['market_cap'] = val

                # Ưu tiên 2: Nếu chưa có, lấy từ Ratio 
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
                # Nếu giá trị max > 10 triệu, có khả năng cao là đơn vị Đồng -> chia cho 1 tỷ
                if s.abs().max() > 10_000_000: 
                    return s / 1_000_000_000
                return s
        return pd.Series(0.0, index=self.raw_reports.index)

    def calculate_metrics(self, fill_nan=True, seasonal_method='additive'):
        """
        Tính toán các chỉ số tài chính
        
        Args:
            fill_nan: Có tự động điền NaN hay không (mặc định: True)
            seasonal_method: Phương pháp seasonal decomposition ('additive' hoặc 'multiplicative')
        """
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

        if not self.ratio_df.empty:
            try:
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

        elif 'dầu khí' in industry2 or 'điện' in industry or 'năng lượng' in industry or 'năng lượng' in industry2:
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

            metrics['FCF (Tỷ)'] = ocf - capex.abs() 
            
            div_cash_paid = self._get_val(['Cổ tức đã trả']).abs()
            metrics['Cổ tức tiền mặt đã trả (Tỷ)'] = div_cash_paid

        else: # Sản xuất / Thương mại
            metrics['Biên LN Gộp (%)'] = safe_div(gross_profit, revenue) * 100
            metrics['Vòng quay kho'] = safe_div(cogs, inventory)
            metrics['Thanh toán hiện hành'] = safe_div(cur_asset, cur_liab)
            metrics['FCF (Tỷ)'] = ocf - capex.abs() 
            
            # Vay nợ
            total_debt = self._get_val(['Vay và nợ thuê tài chính ngắn hạn']) + self._get_val(['Vay và nợ thuê tài chính dài hạn'])
            cash = self._get_val(['Tiền và tương đương tiền'])
            metrics['Vay ròng/Vốn chủ (Lần)'] = safe_div(total_debt - cash, equity)

        # Đảm bảo FCF luôn ở đơn vị Tỷ
        if 'FCF (Tỷ)' in metrics.columns:
            fcf_series = metrics['FCF (Tỷ)']
            # Nếu có giá trị tuyệt đối > 10000 (đã là đơn vị đồng) thì chia cho 1 tỷ
            if fcf_series.abs().max() > 10000:
                metrics['FCF (Tỷ)'] = fcf_series / 1_000_000_000

        # ====== XỬ LÝ NaN BẰNG SEASONAL DECOMPOSITION ======
        if fill_nan and not metrics.empty:
            print("\n--- Đang xử lý giá trị NaN bằng Seasonal Decomposition... ---")
            handler = SeasonalNaNHandler(period=4)  # Chu kỳ theo quý
            
            # Đếm số NaN trước khi xử lý
            nan_before = metrics.isna().sum().sum()
            
            if nan_before > 0:
                print(f"Phát hiện {nan_before} giá trị NaN trong dữ liệu")
                
                # Áp dụng seasonal decomposition cho tất cả các cột
                metrics = handler.fill_dataframe(metrics, method=seasonal_method)
                
                # Đếm số NaN sau khi xử lý
                nan_after = metrics.isna().sum().sum()
                print(f"Còn lại {nan_after} giá trị NaN sau khi xử lý")
                
                if nan_before > nan_after:
                    print(f"✓ Đã điền thành công {nan_before - nan_after} giá trị NaN")

        self.final_metrics = metrics.round(2).sort_index()
        return self.final_metrics

    # --- DISPLAY ---
    def visualize_stock_price(self):
        """Vẽ biểu đồ giá lịch sử với các chỉ báo kỹ thuật và volume"""
        if self.price_history.empty: 
            print("Không có dữ liệu giá lịch sử")
            return
            
        df = self.price_history.sort_index().copy()
        
        # Tính các đường trung bình động
        df['SMA20'] = df['close'].rolling(20).mean()
        df['SMA50'] = df['close'].rolling(50).mean()
        df['SMA200'] = df['close'].rolling(200).mean()
        
        # Tạo figure với 2 subplot (Giá và Volume)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Subplot 1: Biểu đồ giá + chỉ báo
        ax1.plot(df.index, df['close'], label='Giá đóng cửa', color='#1f77b4', linewidth=1.5)
        ax1.plot(df.index, df['SMA20'], label='SMA20', linestyle='--', color='#ff7f0e', alpha=0.8, linewidth=1.2)
        ax1.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='#2ca02c', alpha=0.8, linewidth=1.2)
        ax1.plot(df.index, df['SMA200'], label='SMA200', linestyle='--', color='#d62728', alpha=0.7, linewidth=1.2)
        
        ax1.set_title(f'LỊCH SỬ GIÁ {self.symbol} 1 năm giao dịch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Giá (VNĐ)', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        # Subplot 2: Volume
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Khối lượng', fontsize=11)
        ax2.set_xlabel('Thời gian', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        # Format trục x
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

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
            
            # Vẽ biểu đồ chuyên biệt theo ngành
            self.visualize_industry_specific(df)
        except: pass
    
    def visualize_industry_specific(self, df):
        """Vẽ biểu đồ phân tích theo ngành nghề cụ thể"""
        industry = self.profile_info.get('industry', '').lower()
        industry2 = self.profile_info.get('industry2', '').lower()
        plot_data = df.tail(8)
        
        # NGÂN HÀNG
        if 'ngân hàng' in industry or 'bank' in industry:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.symbol} - PHÂN TÍCH NGÂN HÀNG', fontsize=14, fontweight='bold')
            
            # LDR (Loan to Deposit Ratio)
            if 'LDR (%)' in plot_data.columns:
                axes[0, 0].plot(plot_data.index, plot_data['LDR (%)'], marker='o', color='#1f77b4', linewidth=2)
                axes[0, 0].axhline(85, color='green', linestyle='--', alpha=0.5, label='Ngưỡng an toàn 85%')
                axes[0, 0].set_title('Tỷ lệ Cho vay/Huy động (LDR)')
                axes[0, 0].set_ylabel('LDR (%)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # NIM (Net Interest Margin)
            if 'NIM (%)' in plot_data.columns:
                axes[0, 1].plot(plot_data.index, plot_data['NIM (%)'], marker='s', color='#ff7f0e', linewidth=2)
                axes[0, 1].set_title('Biên lãi suất ròng (NIM)')
                axes[0, 1].set_ylabel('NIM (%)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # CIR (Cost to Income Ratio)
            if 'CIR (%)' in plot_data.columns:
                axes[1, 0].plot(plot_data.index, plot_data['CIR (%)'], marker='^', color='#2ca02c', linewidth=2)
                axes[1, 0].axhline(40, color='red', linestyle='--', alpha=0.5, label='Ngưỡng hiệu quả 40%')
                axes[1, 0].set_title('Tỷ lệ Chi phí/Thu nhập (CIR)')
                axes[1, 0].set_ylabel('CIR (%)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # ROE
            if 'ROE (Quý) (%)' in plot_data.columns:
                axes[1, 1].plot(plot_data.index, plot_data['ROE (Quý) (%)'], marker='D', color='#d62728', linewidth=2)
                axes[1, 1].set_title('ROE theo Quý')
                axes[1, 1].set_ylabel('ROE (%)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # BẤT ĐỘNG SẢN
        elif 'bất động sản' in industry:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.symbol} - PHÂN TÍCH BẤT ĐỘNG SẢN', fontsize=14, fontweight='bold')
            
            # Hàng tồn kho
            if 'Hàng tồn kho (Tỷ)' in plot_data.columns:
                axes[0, 0].bar(range(len(plot_data)), plot_data['Hàng tồn kho (Tỷ)'], color='#1f77b4', alpha=0.7)
                axes[0, 0].set_title('Hàng tồn kho (Dự án BĐS)')
                axes[0, 0].set_ylabel('Tỷ đồng')
                axes[0, 0].set_xticks(range(len(plot_data)))
                axes[0, 0].set_xticklabels(plot_data.index, rotation=45)
                axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Người mua trả trước
            if 'Người mua trả trước (Tỷ)' in plot_data.columns:
                axes[0, 1].bar(range(len(plot_data)), plot_data['Người mua trả trước (Tỷ)'], color='#ff7f0e', alpha=0.7)
                axes[0, 1].set_title('Người mua trả tiền trước')
                axes[0, 1].set_ylabel('Tỷ đồng')
                axes[0, 1].set_xticks(range(len(plot_data)))
                axes[0, 1].set_xticklabels(plot_data.index, rotation=45)
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Tỷ lệ Trả trước/Tồn kho
            if 'Tỷ lệ Trả trước/Tồn kho (%)' in plot_data.columns:
                axes[1, 0].plot(plot_data.index, plot_data['Tỷ lệ Trả trước/Tồn kho (%)'], marker='o', color='#2ca02c', linewidth=2)
                axes[1, 0].set_title('Tỷ lệ Trả trước/Tồn kho')
                axes[1, 0].set_ylabel('%')
                axes[1, 0].grid(True, alpha=0.3)
            
            # FCF
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
            plt.show()
        
        # BẢO HIỂM
        elif 'bảo hiểm' in industry:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.symbol} - PHÂN TÍCH BẢO HIỂM', fontsize=14, fontweight='bold')
            
            # Tỷ trọng đầu tư
            if 'Đầu tư/Tổng TS (%)' in plot_data.columns:
                axes[0, 0].plot(plot_data.index, plot_data['Đầu tư/Tổng TS (%)'], marker='o', color='#1f77b4', linewidth=2)
                axes[0, 0].set_title('Tỷ trọng Đầu tư/Tổng tài sản')
                axes[0, 0].set_ylabel('%')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Lợi suất đầu tư
            if 'Lợi suất đầu tư (%)' in plot_data.columns:
                axes[0, 1].plot(plot_data.index, plot_data['Lợi suất đầu tư (%)'], marker='s', color='#ff7f0e', linewidth=2)
                axes[0, 1].set_title('Lợi suất đầu tư')
                axes[0, 1].set_ylabel('%')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Tỷ lệ chi phí hoạt động
            if 'Tỷ lệ chi phí HĐ (%)' in plot_data.columns:
                axes[1, 0].plot(plot_data.index, plot_data['Tỷ lệ chi phí HĐ (%)'], marker='^', color='#2ca02c', linewidth=2)
                axes[1, 0].set_title('Tỷ lệ chi phí hoạt động')
                axes[1, 0].set_ylabel('%')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Đòn bẩy tài chính
            if 'Đòn bẩy tài chính (Lần)' in plot_data.columns:
                axes[1, 1].plot(plot_data.index, plot_data['Đòn bẩy tài chính (Lần)'], marker='D', color='#d62728', linewidth=2)
                axes[1, 1].set_title('Đòn bẩy tài chính')
                axes[1, 1].set_ylabel('Lần')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        

        # DẦU KHÍ / ĐIỆN
        elif 'dầu khí' in industry2 or 'điện' in industry:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.symbol} - PHÂN TÍCH NĂNG LƯỢNG', fontsize=14, fontweight='bold')
            
            # EBITDA
            if 'EBITDA (Tỷ)' in plot_data.columns:
                axes[0, 0].bar(range(len(plot_data)), plot_data['EBITDA (Tỷ)'], color='#1f77b4', alpha=0.7)
                axes[0, 0].set_title('EBITDA')
                axes[0, 0].set_ylabel('Tỷ đồng')
                axes[0, 0].set_xticks(range(len(plot_data)))
                axes[0, 0].set_xticklabels(plot_data.index, rotation=45)
                axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Biên EBITDA
            if 'Biên EBITDA (%)' in plot_data.columns:
                axes[0, 1].plot(plot_data.index, plot_data['Biên EBITDA (%)'], marker='o', color='#ff7f0e', linewidth=2)
                axes[0, 1].set_title('Biên EBITDA')
                axes[0, 1].set_ylabel('%')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Vay ròng/EBITDA
            if 'Vay ròng/EBITDA (Lần)' in plot_data.columns:
                axes[1, 0].plot(plot_data.index, plot_data['Vay ròng/EBITDA (Lần)'], marker='s', color='#2ca02c', linewidth=2)
                axes[1, 0].axhline(3, color='red', linestyle='--', alpha=0.5, label='Ngưỡng an toàn 3x')
                axes[1, 0].set_title('Vay ròng/EBITDA')
                axes[1, 0].set_ylabel('Lần')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # FCF
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
            plt.show()
        
        # SẢN XUẤT / THƯƠNG MẠI (Mặc định)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{self.symbol} - PHÂN TÍCH SẢN XUẤT/THƯƠNG MẠI', fontsize=14, fontweight='bold')
            
            # Biên lợi nhuận
            has_gross = 'Biên LN Gộp (%)' in plot_data.columns
            has_net = 'Biên LN Ròng (%)' in plot_data.columns
            
            if has_gross or has_net:
                if has_gross:
                    axes[0, 0].plot(plot_data.index, plot_data['Biên LN Gộp (%)'], marker='o', color='#1f77b4', linewidth=2, label='Biên LN Gộp')
                if has_net:
                    axes[0, 0].plot(plot_data.index, plot_data['Biên LN Ròng (%)'], marker='s', color='#ff7f0e', linewidth=2, label='Biên LN Ròng')
                axes[0, 0].set_title('Biên lợi nhuận')
                axes[0, 0].set_ylabel('%')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Vòng quay kho
            if 'Vòng quay kho' in plot_data.columns:
                axes[0, 1].bar(range(len(plot_data)), plot_data['Vòng quay kho'], color='#2ca02c', alpha=0.7)
                axes[0, 1].set_title('Vòng quay hàng tồn kho')
                axes[0, 1].set_ylabel('Lần')
                axes[0, 1].set_xticks(range(len(plot_data)))
                axes[0, 1].set_xticklabels(plot_data.index, rotation=45)
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Thanh toán hiện hành
            if 'Thanh toán hiện hành' in plot_data.columns:
                axes[1, 0].plot(plot_data.index, plot_data['Thanh toán hiện hành'], marker='^', color='#d62728', linewidth=2)
                axes[1, 0].axhline(1, color='red', linestyle='--', alpha=0.5, label='Ngưỡng an toàn 1.0')
                axes[1, 0].set_title('Khả năng thanh toán hiện hành')
                axes[1, 0].set_ylabel('Lần')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # FCF
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
            plt.show()

    def display_price_history(self, num_days=30):
        """Hiển thị bảng giá lịch sử chi tiết"""
        if self.price_history.empty:
            print("Không có dữ liệu giá lịch sử")
            return
        
        df = self.price_history.sort_index(ascending=False).head(num_days).copy()
        
        # Tạo bảng hiển thị
        display_df = pd.DataFrame({
            'Ngày': df.index.strftime('%d/%m/%Y'),
            'Mở cửa': df['open'].apply(lambda x: f"{x:,.2f}"),
            'Cao nhất': df['high'].apply(lambda x: f"{x:,.2f}"),
            'Thấp nhất': df['low'].apply(lambda x: f"{x:,.2f}"),
            'Đóng cửa': df['close'].apply(lambda x: f"{x:,.2f}"),
            'Khối lượng': df['volume'].apply(lambda x: f"{int(x):,}")
        })
        
        print(f"\n>>> LỊCH SỬ GIÁ ({num_days} PHIÊN GẦN NHẤT):")
        print("-" * 100)
        print(display_df.to_string(index=False))
        print("-" * 100)

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

        # Hiển thị thông tin chi tiết công ty
        print(f"\n>>> THÔNG TIN CÔNG TY:")
        
        # Ưu tiên hiển thị organ_name (tên đầy đủ), nếu không có thì dùng short_name
        company_name = info.get('organ_name', '') or info.get('short_name', '')
        if company_name:
            print(f"Tên công ty: {company_name}")
        print(f"Ngành: {info.get('industry', 'N/A')}")
        if info.get('industry2'):
            print(f"Phân ngành: {info.get('industry2')}")
        print(f"Sàn giao dịch: {info.get('exchange', 'N/A')}")
        
        if info.get('website'):
            print(f"Website: {info.get('website')}")
        
        if info.get('established_year'):
            print(f"Năm thành lập: {info.get('established_year')}")
        
        employees = info.get('no_employees', 0)
        if employees > 0:
            print(f"Số lượng nhân viên: {employees:,} người")
        
        shareholders = info.get('no_shareholders', 0)
        if shareholders > 0:
            print(f"Số lượng cổ đông: {shareholders:,}")
        
        foreign = info.get('foreign_percent', 0)
        if foreign > 0:
            print(f"Tỷ lệ sở hữu nước ngoài: {foreign:.2f}%")
        
        outstanding = info.get('outstanding_share', 0)
        if outstanding > 0:
            print(f"Cổ phiếu lưu hành: {outstanding:,.0f} CP")
        
        charter = info.get('charter_capital', 0)
        if charter > 0:
            # Chuyển đổi đơn vị (thường là triệu đồng)
            if charter < 1000000:  # Nếu < 1 triệu tỷ thì là đơn vị triệu
                print(f"Vốn điều lệ: {charter:,.0f} triệu VNĐ ({charter/1000:,.2f} tỷ VNĐ)")
            else:  # Đã là đơn vị đồng
                print(f"Vốn điều lệ: {charter/1_000_000_000:,.2f} tỷ VNĐ")
        
        print(f"\nGiá hiện tại: {info.get('price', 0):,} VNĐ ({info.get('pct_change', 0)*100:.2f}%)")
        print(f"Vốn hóa thị trường: {cap_str}")
        
        if info['officers']:
            print(f"\n>>> BAN LÃNH ĐẠO:")
            for p in info['officers'][:5]: print(f" - {p.get('officer_name')} ({p.get('officer_position', 'N/A')})")
        
        self.display_text_table(info['news'], 'date_str', 'news_title', 'news_source_link', "TIN TỨC")
        self.display_text_table(info['events'], 'date_str', 'display_name', 'event_link', "SỰ KIỆN DOANH NGHIỆP")

        # Hiển thị bảng giá lịch sử
        self.display_price_history(num_days=30)

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