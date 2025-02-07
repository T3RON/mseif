import google.generativeai as genai
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import ssl
import certifi
import os
from tqdm import tqdm
import sys
from colorama import init, Fore, Back, Style
import itertools
import threading
from bs4 import BeautifulSoup
import requests
import winsound
from pygame import mixer

# اضافه کردن در ابتدای فایل
init(autoreset=True)  # برای رنگی کردن متن‌ها

# حذف تنظیم متغیر محیطی (نیازی نیست)
# os.environ['GOOGLE_API_KEY'] = '...'

class GeminiForexBot:
    @staticmethod
    def get_available_symbols():
        """دریافت لیست جفت ارزهای موجود"""
        try:
            if not mt5.initialize():
                logging.error("Failed to initialize MT5")
                return []
            
            # دریافت همه نمادها
            symbols = mt5.symbols_get()
            if not symbols:
                logging.error("No symbols found")
                return []
            
            # فیلتر کردن فقط جفت ارزها
            forex_pairs = []
            for symbol in symbols:
                if symbol.path.startswith("Forex"):
                    # دریافت اطلاعات قیمت
                    tick = mt5.symbol_info_tick(symbol.name)
                    if tick:
                        forex_pairs.append({
                            'name': symbol.name,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'spread': round((tick.ask - tick.bid) * 100000, 1)  # اسپرد به پیپ
                        })
            
            mt5.shutdown()
            return forex_pairs
            
        except Exception as e:
            logging.error(f"Error getting symbols: {e}")
            return []

    @staticmethod
    def select_symbol():
        """انتخاب جفت ارز توسط کاربر"""
        try:
            print(f"\n{Fore.CYAN}=== Available Currency Pairs ==={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}{'Symbol':<10} {'Bid':<10} {'Ask':<10} {'Spread':<10}{Style.RESET_ALL}")
            print("="*40)
            
            pairs = GeminiForexBot.get_available_symbols()
            for i, pair in enumerate(pairs, 1):
                print(f"{i:2}. {pair['name']:<10} {pair['bid']:<10.5f} {pair['ask']:<10.5f} {pair['spread']:<10}")
            
            while True:
                try:
                    choice = input(f"\n{Fore.GREEN}Enter the number of your desired currency pair (1-{len(pairs)}): {Style.RESET_ALL}")
                    index = int(choice) - 1
                    if 0 <= index < len(pairs):
                        return pairs[index]['name']
                    else:
                        print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
                    
        except Exception as e:
            logging.error(f"Error in symbol selection: {e}")
            return "EURUSD"  # جفت ارز پیش‌فرض

    def __init__(self, api_key, mt5_login, mt5_password, mt5_server="", symbol=None, risk_percent=2):
        try:
            if symbol is None:
                symbol = self.select_symbol()
            
            self.symbol = symbol
            self.risk_percent = risk_percent
            self.mt5_login = mt5_login
            self.mt5_password = mt5_password
            self.mt5_server = mt5_server
            
            self.sound_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.mp3")
            mixer.init()
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'forex_bot_{datetime.now().strftime("%Y%m%d")}.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            logging.info(f"{Fore.CYAN}Starting Forex Bot initialization...{Style.RESET_ALL}")
            
            genai.configure(api_key=api_key, transport="rest")
            self.model = genai.GenerativeModel('gemini-pro')
            test_response = self.model.generate_content("Test connection")
            logging.info("Gemini API initialized and tested successfully")
            
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    logging.info(f"Attempting to initialize MetaTrader5 (attempt {attempt + 1}/{max_retries})")
                    mt5.shutdown()
                    time.sleep(1)
                    
                    if not mt5.initialize(
                        login=self.mt5_login,
                        password=self.mt5_password,
                        server=self.mt5_server,
                        timeout=30000
                    ):
                        error_code = mt5.last_error()
                        logging.error(f"MetaTrader5 initialization failed. Error code: {error_code}")
                        if attempt < max_retries - 1:
                            logging.info(f"Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            continue
                        raise Exception(f"MetaTrader5 initialization failed with error code: {error_code}")
                    
                    account_info = mt5.account_info()
                    if account_info is None:
                        raise Exception("Could not get account information")
                    
                    logging.info(f"Successfully connected to MetaTrader5. Account: {account_info.login}")
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to initialize MetaTrader5: {e}")
                    logging.warning(f"Initialization attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                    
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise
    
    def calculate_position_size(self, stop_loss_pips):
        """محاسبه حجم معامله با مشورت هوش مصنوعی"""
        try:
            account_info = mt5.account_info()
            balance = account_info.balance
            
            context = f"""
            Account Information:
            - Balance: {balance}
            - Risk Percent: {self.risk_percent}%
            - Stop Loss: {stop_loss_pips} pips
            - Current Market Volatility: {self.get_current_volatility()}
            """

            question = "What is the optimal position size considering account balance, risk management, and current market conditions?"

            ai_response = self.ask_ai(question, context)
            
            # پردازش پاسخ AI و استخراج حجم پیشنهادی
            suggested_lot = float(ai_response.split("lot size:")[-1].split()[0])
            
            # محدود کردن حجم معامله به محدوده منطقی
            return min(max(suggested_lot, 0.01), 1.0)
            
        except Exception as e:
            logging.error(f"Error in AI position sizing: {e}")
            return 0.01

    def get_current_volatility(self):
        """محاسبه نوسان‌پذیری فعلی بازار"""
        try:
            data = self.get_market_data()
            return data['Volatility'].iloc[-1]
        except:
            return 0

    def ask_ai(self, question, context):
        """پرسش از هوش مصنوعی برای هر تصمیم‌گیری"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                prompt = f"""As an expert forex trading AI assistant, analyze this context and answer the question.

                Context:
                {context}

                Question:
                {question}

                Please provide a detailed analysis and clear recommendation."""

                # تنظیم SSL برای درخواست
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                ssl_context.check_hostname = True

                response = self.model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                if "SSL" in str(e) or "HTTPSConnectionPool" in str(e):
                    if attempt < max_retries - 1:
                        logging.warning(f"SSL Error in API call (attempt {attempt + 1}/{max_retries}): {e}")
                        time.sleep(retry_delay)
                        continue
                logging.error(f"Error asking AI: {e}")
                return None
        
        # اگر همه تلاش‌ها شکست خورد
        return None

    def validate_trade_conditions(self, signal, data):
        """بررسی شرایط معامله با کمک هوش مصنوعی"""
        try:
            # آماده‌سازی اطلاعات برای AI
            context = f"""
            Current Market Conditions:
            - Signal: {signal}
            - Current Price: {data['close'].iloc[-1]}
            - RSI: {data['RSI'].iloc[-1]}
            - MACD: {data['MACD'].iloc[-1]}
            - Volume: {data['tick_volume'].iloc[-1]}
            - Volatility: {data['Volatility'].iloc[-1]}
            """

            question = "Should we execute this trade based on current market conditions? Consider trend, volume, volatility, and overall market health."

            ai_response = self.ask_ai(question, context)
            
            if "yes" in ai_response.lower():
                return True
            return False

        except Exception as e:
            logging.error(f"Error in AI trade validation: {e}")
            return False

    def get_pip_value(self):
        """محاسبه ارزش هر پیپ"""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            return symbol_info.trade_tick_value
        except:
            return 0.0001  # مقدار پیش‌فرض برای جفت‌ارزهای اصلی
    
    def calculate_technical_indicators(self, data):
        """محاسبه شاخص‌های تکنیکال پیشرفته"""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # میانگین‌های متحرک
        df['SMA5'] = df['close'].rolling(window=5).mean()
        df['SMA10'] = df['close'].rolling(window=10).mean()
        df['SMA20'] = df['close'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Momentum
        df['Momentum'] = df['close'].diff(periods=5)
        
        # Volatility
        df['Volatility'] = df['close'].rolling(window=10).std()
        
        return df
    
    def get_market_data(self):
        """دریافت داده‌های بازار در تایم‌فریم 1 دقیقه"""
        try:
            # دریافت 100 کندل آخر در تایم‌فریم 1 دقیقه
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 100)
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # محاسبه شاخص‌های تکنیکال بیشتر
            df = self.calculate_technical_indicators(df)
            return df
            
        except Exception as e:
            logging.error(f"Error getting market data: {e}")
            return None
    
    def get_forex_news(self):
        """دریافت اخبار مهم فارکس از ForexFactory به صورت رایگان"""
        try:
            # دریافت صفحه تقویم اقتصادی
            url = "https://www.forexfactory.com/calendar"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # یافتن رویدادهای امروز
            calendar_table = soup.find('table', class_='calendar__table')
            events = []
            
            if calendar_table:
                current_date = datetime.now()
                
                for row in calendar_table.find_all('tr', class_='calendar__row'):
                    try:
                        # استخراج زمان
                        time_cell = row.find('td', class_='calendar__time')
                        if time_cell:
                            time_str = time_cell.text.strip()
                            if time_str:
                                try:
                                    event_time = datetime.strptime(f"{current_date.date()} {time_str}", "%Y-%m-%d %H:%M")
                                except:
                                    continue
                                
                                # استخراج ارز
                                currency = row.find('td', class_='calendar__currency')
                                currency = currency.text.strip() if currency else ""
                                
                                # استخراج عنوان خبر
                                title = row.find('td', class_='calendar__event')
                                title = title.text.strip() if title else ""
                                
                                # استخراج اهمیت خبر
                                impact = row.find('td', class_='calendar__impact')
                                if impact:
                                    impact_spans = impact.find_all('span', class_='impact')
                                    if len(impact_spans) == 3:
                                        impact = "HIGH"
                                    elif len(impact_spans) == 2:
                                        impact = "MEDIUM"
                                    else:
                                        impact = "LOW"
                                
                                # استخراج مقادیر
                                actual = row.find('td', class_='calendar__actual')
                                actual = actual.text.strip() if actual else None
                                
                                forecast = row.find('td', class_='calendar__forecast')
                                forecast = forecast.text.strip() if forecast else None
                                
                                previous = row.find('td', class_='calendar__previous')
                                previous = previous.text.strip() if previous else None
                                
                                # اضافه کردن به لیست اخبار
                                events.append({
                                    'time': event_time,
                                    'currency': currency,
                                    'title': title,
                                    'impact': impact,
                                    'actual': actual,
                                    'forecast': forecast,
                                    'previous': previous
                                })
                                
                    except Exception as e:
                        logging.warning(f"Error parsing news row: {e}")
                        continue
            
            # مرتب‌سازی بر اساس زمان
            events.sort(key=lambda x: x['time'])
            
            # لاگ کردن اخبار مهم
            for event in events:
                if event['impact'] == "HIGH":
                    logging.info(f"""
                    🔴 High Impact News:
                    Currency: {event['currency']}
                    Event: {event['title']}
                    Time: {event['time'].strftime('%H:%M')}
                    Impact: {event['impact']}
                    Actual: {event['actual']}
                    Forecast: {event['forecast']}
                    Previous: {event['previous']}
                    """)
            
            return events
            
        except Exception as e:
            logging.error(f"Error fetching forex news: {e}")
            return []
    
    def analyze_price_action(self, data):
        """تحلیل پرایس اکشن با استفاده از الگوهای کندل استیک و سطوح کلیدی"""
        try:
            if data is None or len(data) < 2:
                logging.error("Insufficient data for price action analysis")
                return {
                    'patterns': {},
                    'key_levels': [],
                    'volume_analysis': {
                        'volume_surge': False,
                        'volume_trend': 'neutral'
                    },
                    'breakout_analysis': []
                }

            df = data.copy()
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # تحلیل الگوهای کندل استیک
            patterns = {}
            try:
                patterns = {
                    'pin_bar': self.is_pin_bar(current_candle) if hasattr(self, 'is_pin_bar') else None,
                    'engulfing': self.is_engulfing_pattern(current_candle, prev_candle) if hasattr(self, 'is_engulfing_pattern') else None,
                    'doji': self.is_doji(current_candle) if hasattr(self, 'is_doji') else None,
                    'hammer': self.is_hammer(current_candle) if hasattr(self, 'is_hammer') else None,
                    'shooting_star': self.is_shooting_star(current_candle) if hasattr(self, 'is_shooting_star') else None
                }
            except Exception as e:
                logging.error(f"Error analyzing candlestick patterns: {e}")
                patterns = {}
            
            # شناسایی سطوح کلیدی
            try:
                key_levels = self.find_key_levels(df) if hasattr(self, 'find_key_levels') else []
            except Exception as e:
                logging.error(f"Error finding key levels: {e}")
                key_levels = []
            
            # تحلیل حجم
            try:
                volume_analysis = self.analyze_volume(df) if hasattr(self, 'analyze_volume') else {
                    'volume_surge': False,
                    'volume_trend': 'neutral'
                }
            except Exception as e:
                logging.error(f"Error analyzing volume: {e}")
                volume_analysis = {
                    'volume_surge': False,
                    'volume_trend': 'neutral'
                }
            
            # تحلیل شکست‌ها
            try:
                breakout_analysis = self.analyze_breakouts(df, key_levels) if hasattr(self, 'analyze_breakouts') else []
            except Exception as e:
                logging.error(f"Error analyzing breakouts: {e}")
                breakout_analysis = []
            
            return {
                'patterns': patterns,
                'key_levels': key_levels,
                'volume_analysis': volume_analysis,
                'breakout_analysis': breakout_analysis
            }
            
        except Exception as e:
            logging.error(f"Error in price action analysis: {e}")
            return {
                'patterns': {},
                'key_levels': [],
                'volume_analysis': {
                    'volume_surge': False,
                    'volume_trend': 'neutral'
                },
                'breakout_analysis': []
            }

    def is_pin_bar(self, candle):
        """تشخیص الگوی پین بار"""
        body_size = abs(candle['open'] - candle['close'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        
        # پین بار صعودی
        if lower_wick > (body_size * 2) and upper_wick < (body_size * 0.5):
            return {'type': 'bullish', 'strength': lower_wick / body_size if body_size > 0 else 0}
        
        # پین بار نزولی
        elif upper_wick > (body_size * 2) and lower_wick < (body_size * 0.5):
            return {'type': 'bearish', 'strength': upper_wick / body_size if body_size > 0 else 0}
        
        return None

    def is_engulfing_pattern(self, current, previous):
        """تشخیص الگوی انگالفینگ"""
        current_body = abs(current['close'] - current['open'])
        prev_body = abs(previous['close'] - previous['open'])
        
        # انگالفینگ صعودی
        if (current['open'] < previous['close'] and 
            current['close'] > previous['open'] and 
            current_body > prev_body and
            current['close'] > current['open']):
            return {'type': 'bullish', 'strength': current_body / prev_body}
        
        # انگالفینگ نزولی
        elif (current['open'] > previous['close'] and 
              current['close'] < previous['open'] and 
              current_body > prev_body and
              current['close'] < current['open']):
            return {'type': 'bearish', 'strength': current_body / prev_body}
        
        return None

    def find_key_levels(self, data):
        """شناسایی سطوح کلیدی حمایت و مقاومت"""
        df = data.copy()
        levels = []
        
        # یافتن نقاط سوینگ
        for i in range(2, len(df) - 2):
            # سوینگ های بالا
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                levels.append({
                    'price': df['high'].iloc[i],
                    'type': 'resistance',
                    'strength': self.calculate_level_strength(df, df['high'].iloc[i])
                })
            
            # سوینگ های پایین
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                levels.append({
                    'price': df['low'].iloc[i],
                    'type': 'support',
                    'strength': self.calculate_level_strength(df, df['low'].iloc[i])
                })
        
        return levels

    def calculate_level_strength(self, data, level_price):
        """محاسبه قدرت سطح حمایت/مقاومت"""
        touches = 0
        bounces = 0
        
        for i in range(len(data)):
            # بررسی برخورد قیمت با سطح
            if abs(data['high'].iloc[i] - level_price) < 0.0010 or abs(data['low'].iloc[i] - level_price) < 0.0010:
                touches += 1
                
                # بررسی برگشت قیمت
                if i < len(data) - 1:
                    if level_price > data['close'].iloc[i] and data['close'].iloc[i+1] < data['close'].iloc[i]:
                        bounces += 1
                    elif level_price < data['close'].iloc[i] and data['close'].iloc[i+1] > data['close'].iloc[i]:
                        bounces += 1
        
        return {
            'touches': touches,
            'bounces': bounces,
            'reliability': (bounces / touches) if touches > 0 else 0
        }

    def analyze_volume(self, data):
        """تحلیل حجم معاملات"""
        df = data.copy()
        current_volume = df['tick_volume'].iloc[-1]
        avg_volume = df['tick_volume'].rolling(window=20).mean().iloc[-1]
        
        volume_increase = current_volume > avg_volume * 1.5
        volume_trend = 'increasing' if df['tick_volume'].iloc[-3:].is_monotonic_increasing else 'decreasing'
        
        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_surge': volume_increase,
            'volume_trend': volume_trend
        }

    def analyze_breakouts(self, data, key_levels):
        """تحلیل شکست‌های قیمتی"""
        df = data.copy()
        current_price = df['close'].iloc[-1]
        breakouts = []
        
        for level in key_levels:
            if level['type'] == 'resistance':
                # بررسی شکست مقاومت
                if (current_price > level['price'] and 
                    df['close'].iloc[-2] <= level['price']):
                    breakouts.append({
                        'type': 'bullish',
                        'level': level['price'],
                        'strength': level['strength']['reliability']
                    })
            else:
                # بررسی شکست حمایت
                if (current_price < level['price'] and 
                    df['close'].iloc[-2] >= level['price']):
                    breakouts.append({
                        'type': 'bearish',
                        'level': level['price'],
                        'strength': level['strength']['reliability']
                    })
        
        return breakouts
    
    def validate_signal_with_price_action(self, signal_type, pa_analysis):
        """تایید سیگنال با تحلیل پرایس اکشن"""
        if signal_type == "WAIT":
            return True
            
        confirmations = 0
        
        # بررسی الگوهای کندل استیک
        for pattern_name, pattern_data in pa_analysis['patterns'].items():
            if pattern_data:
                if pattern_data['type'] == signal_type.lower():
                    confirmations += 1
                    break
        
        # بررسی شکست‌ها
        for breakout in pa_analysis['breakout_analysis']:
            if breakout['type'] == signal_type.lower():
                confirmations += 1
                break
        
        # بررسی حجم
        if pa_analysis['volume_analysis']['volume_surge']:
            confirmations += 1
        elif pa_analysis['volume_analysis']['volume_trend'] == 'increasing':
            confirmations += 0.5
        
        # بررسی سطوح کلیدی
        for level in pa_analysis['key_levels']:
            if (signal_type == 'UP' and level['type'] == 'support') or \
               (signal_type == 'DOWN' and level['type'] == 'resistance'):
                if level['strength']['reliability'] > 0.5:
                    confirmations += 1
                    break
        
        # کاهش آستانه تایید به 1.5 (قبلاً 2 بود)
        return confirmations >= 1.5
    
    def analyze_with_gemini(self, data):
        """تحلیل ترکیبی پرایس اکشن و هوش مصنوعی"""
        try:
            if data is None:
                logging.error("No market data available")
                return None

            # تحلیل پرایس اکشن
            pa_analysis = self.analyze_price_action(data)
            if pa_analysis is None:
                pa_analysis = {
                    'patterns': {},
                    'key_levels': [],
                    'volume_analysis': {
                        'volume_surge': False,
                        'volume_trend': 'neutral'
                    },
                    'breakout_analysis': []
                }
            
            # دریافت اخبار با مدیریت خطا
            try:
                news_data = self.get_market_news()
                if not news_data:
                    news_data = {'ForexFactory': []}
            except Exception as e:
                logging.error(f"Error getting news: {e}")
                news_data = {'ForexFactory': []}
            
            current_time = datetime.now()
            
            # تحلیل شرایط اخبار
            news_conditions = {
                'high_impact_news': False,
                'market_sentiment': 'neutral',
                'news_summary': []
            }
            
            if news_data and 'ForexFactory' in news_data and news_data['ForexFactory']:
                for event in news_data['ForexFactory']:
                    time_diff = (event['time'] - current_time).total_seconds() / 60
                    if event['impact'] == 'HIGH' and 0 <= time_diff <= 30:
                        news_conditions['high_impact_news'] = True
                        news_conditions['news_summary'].append(f"HIGH IMPACT NEWS: {event['currency']} - {event['title']}")
            
            # تحلیل تکنیکال
            try:
                last_candle = data.iloc[-1]
                current_price = last_candle['close']
                
                # بررسی وجود شاخص‌های تکنیکال
                required_indicators = ['RSI', 'MACD', 'Signal_Line']
                for indicator in required_indicators:
                    if indicator not in last_candle:
                        logging.error(f"Missing indicator: {indicator}")
                        return None
                
                # ساخت متن تحلیل برای AI
        market_context = f"""
                Detailed Market Analysis for {self.symbol}:
                
                Current Price: {current_price:.5f}
                
                PRICE ACTION ANALYSIS:
                Candlestick Patterns:
                {self.format_patterns(pa_analysis['patterns'])}
                
                Key Levels:
                {self.format_key_levels(pa_analysis['key_levels'])}
                
                Volume Analysis:
                - Current Volume vs Average: {'High' if pa_analysis['volume_analysis']['volume_surge'] else 'Normal'}
                - Volume Trend: {pa_analysis['volume_analysis']['volume_trend']}
                
                Breakout Analysis:
                {self.format_breakouts(pa_analysis['breakout_analysis'])}
                
                Technical Indicators:
                - RSI(14): {last_candle['RSI']:.2f}
                - MACD: {last_candle['MACD']:.5f}
                - Signal Line: {last_candle['Signal_Line']:.5f}
                
                MARKET NEWS AND CONDITIONS:
                {'⚠️ HIGH IMPACT NEWS ALERT ⚠️' if news_conditions['high_impact_news'] else '✅ No High Impact News'}
                {chr(10).join(news_conditions['news_summary']) if news_conditions['news_summary'] else 'No significant news'}
                """
                
                prompt = f"""You are an expert forex analyst. Analyze this market data to provide a trading signal.
                Focus on clear directional movements and provide more frequent signals when conditions align.
                
        {market_context}
                
                Rules:
                1. Look for clear directional movements
                2. Volume should preferably confirm the movement
                3. Consider key support/resistance levels
                4. If there is high impact news in next 30 minutes, respond with WAIT
                5. Minimum confidence should be 65% for signals
                6. Signal should be supported by at least one strong technical factor
                
                Example Response:
                SIGNAL: UP
                CONFIDENCE: 75
                EXPIRY: 2
                ANALYSIS: Strong bullish momentum with RSI showing oversold conditions and positive MACD crossover.
                
                Or for waiting:
                SIGNAL: WAIT
                CONFIDENCE: 0
                EXPIRY: 1
                ANALYSIS: Market conditions unclear, waiting for better setup.
                
                Provide your analysis in exactly this format.
        """
        
        response = self.model.generate_content(prompt)
                if response is None:
                    logging.error("No response from AI model")
                    return None
                
                # نمایش تحلیل با فرمت زیبا
                self.display_analysis(response.text, pa_analysis)
                
                parsed_response = self._parse_binary_response(response.text)
                
                if parsed_response:
                    signal_type, confidence, expiry, analysis = parsed_response
                    
                    # اعتبارسنجی نهایی سیگنال
                    if news_conditions['high_impact_news']:
                        logging.warning("Signal ignored due to upcoming high impact news")
                        return None
                        
                    # کاهش آستانه اطمینان به 65%
                    if confidence < 65:
                        logging.info("Signal ignored due to low confidence")
                        return None
                    
                    # تایید سیگنال با پرایس اکشن
                    if not self.validate_signal_with_price_action(signal_type, pa_analysis):
                        logging.info("Signal rejected: Does not align with price action")
                        return None
                    
                    return parsed_response
                else:
                    logging.error("Could not parse AI response")
                    return None
                    
            except Exception as e:
                logging.error(f"Error processing technical analysis: {e}")
                return None
                
        except Exception as e:
            logging.error(f"{Fore.RED}Error in analysis: {e}{Style.RESET_ALL}")
            return None

    def format_patterns(self, patterns):
        """فرمت‌بندی الگوهای کندل استیک"""
        result = []
        for pattern_name, pattern_data in patterns.items():
            if pattern_data:
                result.append(f"- {pattern_name.replace('_', ' ').title()}: {pattern_data['type'].title()} (Strength: {pattern_data['strength']:.2f})")
        return '\n'.join(result) if result else "No significant patterns detected"

    def format_key_levels(self, levels):
        """فرمت‌بندی سطوح کلیدی"""
        result = []
        for level in levels:
            result.append(f"- {level['type'].title()}: {level['price']:.5f} (Reliability: {level['strength']['reliability']:.2f})")
        return '\n'.join(result) if result else "No key levels detected"

    def format_breakouts(self, breakouts):
        """فرمت‌بندی شکست‌های قیمتی"""
        result = []
        for breakout in breakouts:
            result.append(f"- {breakout['type'].title()} breakout at {breakout['level']:.5f} (Strength: {breakout['strength']:.2f})")
        return '\n'.join(result) if result else "No recent breakouts detected"

    def analyze_news_impact(self, news, symbol):
        """تحلیل تاثیر اخبار بر جفت ارز"""
        try:
            current_time = datetime.now()
            relevant_news = []
            news_impact = ""
            
            # تشخیص ارزهای جفت ارز
            base_currency = symbol[:3]
            quote_currency = symbol[3:6]
            
            for news_item in news:
                # بررسی اخبار مربوط به ارزهای جفت ارز
                if news_item['currency'] in [base_currency, quote_currency]:
                    time_diff = (news_item['time'] - current_time).total_seconds() / 60
                    
                    if abs(time_diff) < 30:  # اخبار در 30 دقیقه گذشته یا آینده
                        impact_str = "🔴" if news_item['impact'] == 'HIGH' else "🟡"
                        
                        news_impact += f"""
                        {impact_str} {news_item['currency']} - {news_item['title']}
                        Time: {news_item['time'].strftime('%H:%M')} ({int(time_diff)} min {'ago' if time_diff < 0 else 'ahead'})
                        Impact: {news_item['impact']}
                        """
                        
                        if 'actual' in news_item:
                            news_impact += f"""
                            Actual: {news_item['actual']}
                            Forecast: {news_item['forecast']}
                            Previous: {news_item['previous']}
                            """
                        
                        relevant_news.append(news_item)
            
            if not relevant_news:
                return "No significant news events in the next 30 minutes."
            
            return news_impact
            
        except Exception as e:
            logging.error(f"Error analyzing news impact: {e}")
            return "Error analyzing news impact"
    
    def _parse_binary_response(self, response):
        """تجزیه پاسخ AI به اجزای سیگنال"""
        try:
            # لاگ کردن پاسخ خام برای دیباگ
            logging.debug(f"Raw AI response:\n{response}")
            
            lines = response.split('\n')
            signal = "WAIT"
            confidence = 0
            expiry = 1
            analysis = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if 'SIGNAL:' in line:
                        signal = line.split(':')[1].strip().upper()
                elif 'CONFIDENCE:' in line:
                        confidence = float(line.split(':')[1].strip().replace('%', ''))
                    elif 'EXPIRY:' in line:
                        expiry = int(line.split(':')[1].strip().split()[0])  # فقط عدد را بگیر
                    elif 'ANALYSIS:' in line:
                        analysis = line.split(':', 1)[1].strip()
                except ValueError as ve:
                    logging.warning(f"Error parsing line '{line}': {ve}")
                    continue
            
            # اعتبارسنجی مقادیر
            if signal not in ["UP", "DOWN", "WAIT"]:
                logging.error(f"Invalid signal type: {signal}")
                return None
            
            if not (0 <= confidence <= 100):
                logging.error(f"Invalid confidence value: {confidence}")
                return None
            
            if not (1 <= expiry <= 5):
                logging.error(f"Invalid expiry time: {expiry}")
                return None
            
            # لاگ کردن مقادیر پردازش شده
            logging.info(f"""
            Parsed Binary Options values:
            Signal: {signal}
            Confidence: {confidence}%
            Expiry: {expiry} min
            Analysis: {analysis}
            """)
            
            return signal, confidence, expiry, analysis
            
        except Exception as e:
            logging.error(f"Error parsing Binary Options response: {e}")
            return None
    
    def execute_trade(self, action, confidence, current_price, sl_pips, tp_pips):
        """اجرای معامله با توجه به پیش‌بینی"""
        try:
            if confidence < 65:
                return None
            
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                logging.info("Already have open position, skipping trade")
            return None
            
        deviation = 20
            pip_size = 0.0001
            lot = self.calculate_position_size(sl_pips)
            
            if action == "UP":
                sl_price = current_price - (sl_pips * pip_size)
                tp_price = current_price + (tp_pips * pip_size)
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "price": current_price,
                    "sl": sl_price,
                    "tp": tp_price,
                "deviation": deviation,
                "magic": 234000,
                    "comment": f"Gemini AI M1 Buy ({confidence}%)",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            return mt5.order_send(request)
            
            elif action == "DOWN":
                sl_price = current_price + (sl_pips * pip_size)
                tp_price = current_price - (tp_pips * pip_size)
                
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "price": current_price,
                    "sl": sl_price,
                    "tp": tp_price,
                "deviation": deviation,
                "magic": 234000,
                    "comment": f"Gemini AI M1 Sell ({confidence}%)",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            return mt5.order_send(request)
        
        return None
            
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
        return None
    
    def run(self):
        """اجرای ربات در تایم‌فریم 1 دقیقه"""
        logging.info("Bot started running in 1-minute timeframe...")
        last_candle_minute = None
        last_signal_message = None
        last_condition_check = None
        
        # پاک کردن صفحه
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n{Fore.CYAN}🤖 AI FOREX SIGNAL GENERATOR STARTED{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Analyzing {self.symbol} after each 1-minute candle close{Style.RESET_ALL}\n")
        
        while True:
            try:
                current_time = datetime.now()
                current_minute = current_time.minute
                
                # بررسی شرایط بازار هر 5 دقیقه
                if last_condition_check is None or (current_time - last_condition_check).total_seconds() >= 300:
                    print(f"\n{Fore.CYAN}Analyzing market conditions...{Style.RESET_ALL}")
                    self.analyze_market_conditions()
                    last_condition_check = current_time
                
                if current_minute != last_candle_minute:
                    # پاک کردن خط قبلی و نوشتن وضعیت جدید
                    print(f"\r{Fore.CYAN}⏳ Analyzing candle... {current_time.strftime('%H:%M:%S')}{Style.RESET_ALL}", end="")
                    
                    # صبر کن تا ثانیه به 1 برسد
                    while datetime.now().second < 1:
                        time.sleep(0.1)
                    
                    data = self.get_market_data()
                    
                    if data is None or len(data) < 100:
                        print(f"\r{Fore.RED}❌ Data error. Retrying...{Style.RESET_ALL}", end="")
                        time.sleep(1)
                        continue
                    
                    current_price = mt5.symbol_info_tick(self.symbol).ask
                    signal = self.analyze_with_gemini(data)
                    
                    if signal:
                        signal_type, confidence, expiry, analysis = signal
                        
                        if confidence > 60 and signal_type != "WAIT":
                            # پخش صدای آلرت
                            print('\a')  # صدای بیپ سیستم
                            
                            # ایجاد پیام سیگنال
                            signal_message = f"""
╔══════════════════════════════════════════╗
║ {Fore.YELLOW}🔔 BINARY OPTIONS SIGNAL{Style.RESET_ALL}                  ║
╠══════════════════════════════════════════╣
║ Time: {Fore.CYAN}{current_time.strftime('%H:%M:%S')}{Style.RESET_ALL}                        ║
║ Symbol: {self.symbol}                           ║
║ Signal: {Fore.GREEN if signal_type == 'UP' else Fore.RED}{signal_type}{Style.RESET_ALL}                           ║
║ Price: {current_price:.5f}                      ║
║ Confidence: {confidence}% {'✅' if confidence > 75 else '⚠️'}               ║
║ Expiry: {expiry} min ⏱️                         ║
╚══════════════════════════════════════════╝
"""
                            # اگر پیام قبلی وجود دارد، چند خط بالا برو و پیام جدید را جایگزین کن
                            if last_signal_message:
                                # محاسبه تعداد خطوط پیام قبلی
                                lines_to_clear = last_signal_message.count('\n') + 1
                                # برو بالا و پاک کن
                                print(f"\033[{lines_to_clear}A")  # برو بالا
                                print("\033[J", end="")  # پاک کن تا انتها
                            
                            # چاپ پیام جدید
                            print(signal_message)
                            last_signal_message = signal_message
                        else:
                            # آپدیت وضعیت بدون سیگنال
                            print(f"\r{Fore.YELLOW}🔍 Waiting for valid signals... {current_time.strftime('%H:%M:%S')}{Style.RESET_ALL}", end="")
                    
                    last_candle_minute = current_minute
                
                # انتظار کوتاه
                time.sleep(0.5)
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.RED}❌ Bot stopped by user{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
                time.sleep(1)

    def check_news_impact(self):
        """بررسی اخبار مهم اقتصادی"""
        try:
            # اینجا می‌توانید از یک API اخبار اقتصادی استفاده کنید
            current_time = datetime.now()
            
            # مثال ساده برای جلوگیری از معامله در زمان اخبار مهم
            high_impact_news_times = [
                (current_time.replace(hour=14, minute=30), current_time.replace(hour=15, minute=0)),
                # سایر زمان‌های مهم...
            ]
            
            for start, end in high_impact_news_times:
                if start <= current_time <= end:
                    logging.warning("High impact news time - avoiding trades")
                    return True
            return False
        except Exception as e:
            logging.error(f"Error checking news: {e}")
            return True  # در صورت خطا، از معامله جلوگیری کن

    def manage_open_positions(self):
        """مدیریت معاملات باز با کمک هوش مصنوعی"""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                for position in positions:
                    context = f"""
                    Position Information:
                    - Type: {'Buy' if position.type == mt5.ORDER_TYPE_BUY else 'Sell'}
                    - Open Price: {position.price_open}
                    - Current Price: {mt5.symbol_info_tick(self.symbol).ask}
                    - Profit: {position.profit}
                    - Duration: {(datetime.now() - datetime.fromtimestamp(position.time)).minutes} minutes
                    """

                    question = "Should we modify this position (move stop loss, take profit) or close it based on current conditions?"

                    ai_response = self.ask_ai(question, context)
                    
                    if "close" in ai_response.lower():
                        self.close_position(position)
                    elif "modify" in ai_response.lower():
                        # استخراج مقادیر پیشنهادی برای SL و TP
                        new_sl = self.extract_price_from_response(ai_response, "stop loss")
                        new_tp = self.extract_price_from_response(ai_response, "take profit")
                        if new_sl or new_tp:
                            self.modify_position(position, new_sl=new_sl, new_tp=new_tp)

            return len(positions)
        except Exception as e:
            logging.error(f"Error in AI position management: {e}")
            return 0

    def extract_price_from_response(self, response, price_type):
        """استخراج قیمت از پاسخ AI"""
        try:
            if price_type in response.lower():
                # یافتن عدد بعد از عبارت مورد نظر
                price_str = response.lower().split(price_type)[-1].split()[0]
                return float(price_str)
        except:
            return None

    def analyze_market_sentiment(self):
        """تحلیل احساسات بازار با هوش مصنوعی"""
        try:
            # جمع‌آوری داده‌های مختلف
            technical_data = self.get_market_data()
            news_data = self.get_market_news()
            
            context = f"""
            Market Data:
            Technical Analysis:
            - Current Price: {technical_data['close'].iloc[-1]}
            - RSI: {technical_data['RSI'].iloc[-1]}
            - MACD: {technical_data['MACD'].iloc[-1]}
            - Volume: {technical_data['tick_volume'].iloc[-1]}

            Recent News:
            {self.format_news_for_ai(news_data)}
            """

            question = "What is the overall market sentiment considering both technical and fundamental factors? How strong is this sentiment?"

            sentiment_analysis = self.ask_ai(question, context)
            logging.info(f"AI Market Sentiment Analysis: {sentiment_analysis}")
            
            return sentiment_analysis
            
        except Exception as e:
            logging.error(f"Error in AI sentiment analysis: {e}")
            return None

    def format_news_for_ai(self, news_data):
        """فرمت‌بندی اخبار برای ارسال به AI"""
        try:
            if not news_data or 'ForexFactory' not in news_data:
                return "No recent news available"
            
            formatted_news = []
            for event in news_data['ForexFactory']:
                formatted_news.append(f"""
                - Event: {event['title']}
                  Currency: {event['currency']}
                  Impact: {event['impact']}
                  Time: {event['time'].strftime('%H:%M')}
                """)
            
            return "\n".join(formatted_news)
        except:
            return "Error formatting news"

    def predict_price_movement(self, timeframe_minutes=5):
        """پیش‌بینی حرکت قیمت با هوش مصنوعی"""
        try:
            data = self.get_market_data()
            
            context = f"""
            Recent Price Action:
            - Current Price: {data['close'].iloc[-1]}
            - Previous Candles (last 5):
            {data['close'].iloc[-5:].to_string()}
            
            Technical Indicators:
            - RSI: {data['RSI'].iloc[-1]}
            - MACD: {data['MACD'].iloc[-1]}
            - BB Upper: {data['BB_upper'].iloc[-1]}
            - BB Lower: {data['BB_lower'].iloc[-1]}
            
            Market Sentiment: {self.analyze_market_sentiment()}
            """

            question = f"Based on all available data, predict the price movement for the next {timeframe_minutes} minutes. Include probability percentage and key levels to watch."

            prediction = self.ask_ai(question, context)
            logging.info(f"AI Price Prediction: {prediction}")
            
            return prediction
            
        except Exception as e:
            logging.error(f"Error in AI price prediction: {e}")
            return None

    def get_market_news(self):
        """دریافت اخبار بازار از چندین منبع"""
        try:
            news_sources = {
                'ForexFactory': self.get_forex_news(),  # از تابع قبلی استفاده می‌کنیم
                'Investing': [],
                'FXStreet': []
            }
            
            print(f"\n{Fore.CYAN}{'='*50}")
            print(f"{Fore.YELLOW}📰 MARKET NEWS UPDATE{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}\n")
            
            current_time = datetime.now()
            
            # ForexFactory News
            print(f"{Fore.MAGENTA}🔸 Economic Calendar Events:{Style.RESET_ALL}")
            for event in news_sources['ForexFactory']:
                time_diff = (event['time'] - current_time).total_seconds() / 60
                if -30 < time_diff < 120:  # اخبار 30 دقیقه قبل تا 2 ساعت آینده
                    impact_color = (Fore.RED if event['impact'] == 'HIGH' else 
                                  Fore.YELLOW if event['impact'] == 'MEDIUM' else 
                                  Fore.GREEN)
                    
                    print(f"""
{impact_color}{'🔴' if event['impact'] == 'HIGH' else '🟡' if event['impact'] == 'MEDIUM' else '🟢'} {event['currency']} - {event['title']}{Style.RESET_ALL}
   Time: {event['time'].strftime('%H:%M')} ({int(time_diff)} min {'ago' if time_diff < 0 else 'ahead'})
   Impact: {impact_color}{event['impact']}{Style.RESET_ALL}
   {'Actual: ' + str(event['actual']) if event['actual'] else ''} 
   {'Forecast: ' + str(event['forecast']) if event['forecast'] else ''}
   {'Previous: ' + str(event['previous']) if event['previous'] else ''}
""")
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
            # بررسی شرایط خطرناک برای معامله
            high_impact_soon = any(
                event['impact'] == 'HIGH' and 
                0 <= (event['time'] - current_time).total_seconds() / 60 <= 30 
                for event in news_sources['ForexFactory']
            )
            
            if high_impact_soon:
                print(f"\n{Fore.RED}⚠️ WARNING: High impact news coming in next 30 minutes - Trading not recommended!{Style.RESET_ALL}")
            
            return news_sources
            
        except Exception as e:
            logging.error(f"Error getting market news: {e}")
            return {'ForexFactory': [], 'Investing': [], 'FXStreet': []}

    def display_analysis(self, response_text, pa_analysis):
        """نمایش تحلیل با فرمت زیبا"""
        try:
            print(f"\n{Fore.CYAN}{'='*50}")
            print(f"{Fore.YELLOW}🤖 AI ANALYSIS WITH PRICE ACTION{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}\n")
            
            # تجزیه پاسخ و نمایش رنگی
            lines = response_text.split('\n')
            for line in lines:
                if 'SIGNAL:' in line:
                    signal_value = line.split(':')[1].strip().upper()
                    signal_color = (Fore.GREEN if signal_value == 'UP' else 
                                  Fore.RED if signal_value == 'DOWN' else 
                                  Fore.YELLOW)
                    print(f"{Fore.WHITE}SIGNAL: {signal_color}{signal_value}{Style.RESET_ALL}")
                    
                elif 'CONFIDENCE:' in line:
                    conf_value = float(line.split(':')[1].strip().replace('%', ''))
                    conf_color = (Fore.GREEN if conf_value >= 80 else 
                                Fore.YELLOW if conf_value >= 70 else 
                                Fore.RED)
                    print(f"{Fore.WHITE}CONFIDENCE: {conf_color}{conf_value}%{Style.RESET_ALL}")
                    
                elif 'EXPIRY:' in line:
                    print(f"{Fore.WHITE}{line}{Style.RESET_ALL}")
                    
                elif 'ANALYSIS:' in line:
                    analysis_text = line.split(':', 1)[1].strip()
                    print(f"\n{Fore.MAGENTA}📊 Technical Analysis:{Style.RESET_ALL}")
                    
                    # تقسیم تحلیل به بخش‌های مختلف
                    analysis_points = analysis_text.split('.')
                    for point in analysis_points:
                        if point.strip():
                            # رنگ‌آمیزی کلمات کلیدی
                            point = point.strip()
                            point = point.replace('bullish', f"{Fore.GREEN}bullish{Fore.WHITE}")
                            point = point.replace('bearish', f"{Fore.RED}bearish{Fore.WHITE}")
                            point = point.replace('support', f"{Fore.GREEN}support{Fore.WHITE}")
                            point = point.replace('resistance', f"{Fore.RED}resistance{Fore.WHITE}")
                            point = point.replace('breakout', f"{Fore.YELLOW}breakout{Fore.WHITE}")
                            print(f"• {Fore.WHITE}{point}{Style.RESET_ALL}")
            
            # نمایش الگوهای پرایس اکشن
            if pa_analysis:
                print(f"\n{Fore.MAGENTA}📈 Price Action Patterns:{Style.RESET_ALL}")
                for pattern_name, pattern_data in pa_analysis['patterns'].items():
                    if pattern_data:
                        pattern_color = Fore.GREEN if pattern_data['type'] == 'bullish' else Fore.RED
                        print(f"• {pattern_color}{pattern_name.title()}: {pattern_data['type'].title()} (Strength: {pattern_data['strength']:.2f}){Style.RESET_ALL}")
                
                # نمایش سطوح کلیدی
                if pa_analysis['key_levels']:
                    print(f"\n{Fore.MAGENTA}🎯 Key Levels:{Style.RESET_ALL}")
                    for level in pa_analysis['key_levels']:
                        level_color = Fore.RED if level['type'] == 'resistance' else Fore.GREEN
                        print(f"• {level_color}{level['type'].title()}: {level['price']:.5f} (Reliability: {level['strength']['reliability']:.2f}){Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
        except Exception as e:
            logging.error(f"Error displaying analysis: {e}")
            print(f"\n{Fore.RED}Error displaying analysis: {e}{Style.RESET_ALL}")

    def is_doji(self, candle):
        """تشخیص الگوی دوجی"""
        body_size = abs(candle['open'] - candle['close'])
        total_size = candle['high'] - candle['low']
        
        if total_size == 0:
            return None
            
        body_ratio = body_size / total_size
        
        if body_ratio < 0.1:  # بدنه کمتر از 10% کل شمع
            return {
                'type': 'bullish' if candle['close'] > candle['open'] else 'bearish',
                'strength': 1 - body_ratio
            }
        return None

    def is_hammer(self, candle):
        """تشخیص الگوی چکش"""
        body_size = abs(candle['open'] - candle['close'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_size = candle['high'] - candle['low']
        
        if total_size == 0:
            return None
            
        # بدنه کوچک در بالا و سایه بلند در پایین
        if (lower_wick > (2 * body_size) and 
            upper_wick < (0.1 * total_size) and
            body_size < (0.3 * total_size)):
            return {
                'type': 'bullish',
                'strength': lower_wick / total_size
            }
        return None

    def is_shooting_star(self, candle):
        """تشخیص الگوی ستاره دنباله‌دار"""
        body_size = abs(candle['open'] - candle['close'])
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        total_size = candle['high'] - candle['low']
        
        if total_size == 0:
            return None
            
        # بدنه کوچک در پایین و سایه بلند در بالا
        if (upper_wick > (2 * body_size) and 
            lower_wick < (0.1 * total_size) and
            body_size < (0.3 * total_size)):
            return {
                'type': 'bearish',
                'strength': upper_wick / total_size
            }
        return None

    def is_morning_star(self, candles):
        """تشخیص الگوی ستاره صبحگاهی"""
        if len(candles) < 3:
            return None
            
        first = candles.iloc[-3]  # شمع نزولی بزرگ
        second = candles.iloc[-2]  # شمع کوچک
        third = candles.iloc[-1]   # شمع صعودی بزرگ
        
        # شرایط الگو
        if (first['close'] < first['open'] and                     # شمع اول نزولی
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # شمع دوم کوچک
            third['close'] > third['open'] and                     # شمع سوم صعودی
            third['close'] > (first['open'] + first['close']) / 2):  # بازگشت قیمت
            
            return {
                'type': 'bullish',
                'strength': abs(third['close'] - third['open']) / abs(first['close'] - first['open'])
            }
        return None

    def is_evening_star(self, candles):
        """تشخیص الگوی ستاره شامگاهی"""
        if len(candles) < 3:
            return None
            
        first = candles.iloc[-3]  # شمع صعودی بزرگ
        second = candles.iloc[-2]  # شمع کوچک
        third = candles.iloc[-1]   # شمع نزولی بزرگ
        
        # شرایط الگو
        if (first['close'] > first['open'] and                     # شمع اول صعودی
            abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and  # شمع دوم کوچک
            third['close'] < third['open'] and                     # شمع سوم نزولی
            third['close'] < (first['open'] + first['close']) / 2):  # بازگشت قیمت
            
            return {
                'type': 'bearish',
                'strength': abs(third['close'] - third['open']) / abs(first['close'] - first['open'])
            }
        return None

    def analyze_market_conditions(self):
        """تحلیل شرایط بازار و تنش‌های احتمالی با هوش مصنوعی"""
        try:
            # دریافت داده‌های تکنیکال
            data = self.get_market_data()
            if data is None:
                return None
                
            # محاسبه نوسانات اخیر
            volatility = data['Volatility'].iloc[-1]
            avg_volatility = data['Volatility'].rolling(window=20).mean().iloc[-1]
            
            # محاسبه حجم معاملات
            volume = data['tick_volume'].iloc[-1]
            avg_volume = data['tick_volume'].rolling(window=20).mean().iloc[-1]
            
            # دریافت اخبار مهم
            news_data = self.get_forex_news()
            
            # آماده‌سازی متن برای AI
            market_context = f"""
            Analyze the current market conditions for {self.symbol}:
            
            Technical Conditions:
            - Current Price: {data['close'].iloc[-1]}
            - RSI: {data['RSI'].iloc[-1]}
            - Current Volatility vs Average: {(volatility/avg_volatility):.2f}
            - Current Volume vs Average: {(volume/avg_volume):.2f}
            - MACD: {data['MACD'].iloc[-1]}
            - Signal Line: {data['Signal_Line'].iloc[-1]}
            
            Recent Price Movement:
            - Last 5 candles: {data['close'].iloc[-5:].tolist()}
            - Price change %: {((data['close'].iloc[-1] - data['close'].iloc[-5])/data['close'].iloc[-5]*100):.2f}%
            
            Market News:
            """
            
            # اضافه کردن اخبار مهم به متن
            if news_data:
                for event in news_data:
                    if event['impact'] in ['HIGH', 'MEDIUM']:
                        market_context += f"""
                        - {event['currency']} {event['title']}
                          Impact: {event['impact']}
                          Time: {event['time'].strftime('%H:%M')}
                        """
            
            prompt = f"""As an expert forex analyst, analyze these market conditions and answer the following questions:

            1. Is the market currently stable or volatile for {self.symbol}?
            2. Are there any significant risks or tensions in the market?
            3. Is this a good time to trade considering both technical and news factors?
            4. What is the overall market sentiment (bullish/bearish/neutral)?
            5. What are the key levels to watch?

            Please provide a detailed analysis with clear recommendations."""
            
            # دریافت تحلیل از AI
            response = self.model.generate_content(prompt + "\n\n" + market_context)
            
            if response:
                # نمایش تحلیل با فرمت زیبا
                print(f"\n{Fore.CYAN}{'='*50}")
                print(f"{Fore.YELLOW}🔍 MARKET CONDITIONS ANALYSIS{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*50}\n")
                
                analysis_text = response.text
                
                # رنگ‌آمیزی کلمات کلیدی
                analysis_text = analysis_text.replace('stable', f"{Fore.GREEN}stable{Style.RESET_ALL}")
                analysis_text = analysis_text.replace('volatile', f"{Fore.RED}volatile{Style.RESET_ALL}")
                analysis_text = analysis_text.replace('bullish', f"{Fore.GREEN}bullish{Style.RESET_ALL}")
                analysis_text = analysis_text.replace('bearish', f"{Fore.RED}bearish{Style.RESET_ALL}")
                analysis_text = analysis_text.replace('risk', f"{Fore.RED}risk{Style.RESET_ALL}")
                
                # تقسیم تحلیل به پاراگراف‌ها
                paragraphs = analysis_text.split('\n')
                for para in paragraphs:
                    if para.strip():
                        print(f"{para.strip()}\n")
                
                print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
                
                return response.text
            
            return None
            
        except Exception as e:
            logging.error(f"Error analyzing market conditions: {e}")
            return None

def loading_animation():
    """نمایش انیمیشن در حال جستجو"""
    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    for char in itertools.cycle(chars):
        if not loading_animation.running:
            break
        sys.stdout.write(f'\r{Fore.CYAN}Searching for signals... {char}')
        sys.stdout.flush()
        time.sleep(0.1)

# نمونه استفاده
if __name__ == "__main__":
    try:
        # تنظیمات اتصال
        api_key = "AIzaSyBFZTMkEmOVRabDJgdtqD_78OWApAmvDC8"  # API key جدید
        mt5_login = 7140773  # شماره لاگین حساب متاتریدر
        mt5_password = "a!U4Tmw9"  # رمز عبور حساب متاتریدر 
        mt5_server = "AMarkets-Demo"  # نام سرور بروکر
        
        print("Starting bot initialization...")
        
        # تست اولیه API با خطایابی بیشتر
        try:
            print("1. Initializing Gemini API...")
            genai.configure(api_key=api_key, transport="rest")
            
            # تست با جزئیات بیشتر
            print("   Testing API access...")
            model = genai.GenerativeModel('gemini-pro')
            
            print("   Sending test request...")
            response = model.generate_content("Simple test message")
            
            if response and hasattr(response, 'text'):
                print("✓ Gemini API Test Successful")
                print(f"   Response: {response.text[:50]}...")  # نمایش بخشی از پاسخ
            else:
                raise Exception("Invalid response format")
            
            print("\n2. Setting up MetaTrader5...")
            if mt5.initialize():
                print("✓ MetaTrader5 base initialization successful")
            else:
                print(f"✗ MetaTrader5 initialization failed. Error code: {mt5.last_error()}")
                raise Exception("MetaTrader5 initialization failed")
            
            print("\n3. Attempting to login to MetaTrader5...")
            if mt5.login(login=mt5_login, password=mt5_password, server=mt5_server):
                account_info = mt5.account_info()
                if account_info is not None:
                    print(f"✓ Successfully logged in to MetaTrader5")
                    print(f"Account: {account_info.login}")
                    print(f"Balance: {account_info.balance}")
                    print(f"Equity: {account_info.equity}")
                else:
                    print("✗ Failed to get account info after login")
                    raise Exception("Could not get account information")
            else:
                error = mt5.last_error()
                print(f"✗ Login failed. Error code: {error}")
                raise Exception(f"Login failed with error: {error}")
            
            print("\n4. Creating and starting trading bot...")
            bot = GeminiForexBot(
                api_key=api_key,
                mt5_login=mt5_login,
                mt5_password=mt5_password,
                mt5_server=mt5_server
            )
            print("✓ Bot created successfully")
            
            print("\n5. Starting bot operation...")
bot.run()
            
        except Exception as e:
            print(f"\n✗ Initialization Failed: {str(e)}")
            logging.error(f"Initialization Failed: {e}")
            if mt5.initialize():
                mt5.shutdown()
            raise
        
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        logging.info("Bot stopped by user")
        mt5.shutdown()
    except Exception as e:
        print(f"\nFatal error: {e}")
        logging.error(f"Fatal error: {e}")
        mt5.shutdown()

