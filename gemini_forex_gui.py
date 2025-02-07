 import google.generativeai as genai
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
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
from termcolor import colored
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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
            
            # لیست جفت ارزهای اصلی
            major_pairs = [
                'EURUSDb', 'GBPUSDb', 'USDJPYb', 'USDCHFb', 
                'AUDUSDb', 'USDCADb', 'NZDUSDb',
                'EURGBPb', 'EURJPYb', 'GBPJPYb'
            ]
            
            # فیلتر کردن فقط جفت ارزهای اصلی
            forex_pairs = []
            for symbol in symbols:
                if symbol.name in major_pairs:
                    # دریافت اطلاعات قیمت و تاریخچه
                    rates = mt5.copy_rates_from_pos(symbol.name, mt5.TIMEFRAME_M1, 0, 100)
                    if rates is not None:
                        df = pd.DataFrame(rates)
                        
                        # محاسبه شاخص‌های تکنیکال
                        df['SMA20'] = df['close'].rolling(window=20).mean()
                        df['Volatility'] = df['close'].rolling(window=10).std()
                        df['ADX'] = GeminiForexBot.calculate_adx(df)
                        
                        # ارزیابی شرایط بازار
                        last_price = df['close'].iloc[-1]
                        sma20 = df['SMA20'].iloc[-1]
                        volatility = df['Volatility'].iloc[-1]
                        adx = df['ADX'].iloc[-1]
                        
                        # محاسبه قدرت روند
                        trend_strength = 'strong' if adx > 25 else 'weak'
                        trend_direction = 'up' if last_price > sma20 else 'down'
                        
                        forex_pairs.append({
                            'name': symbol.name,
                            'bid': df['close'].iloc[-1],
                            'ask': df['close'].iloc[-1],
                            'trend_strength': trend_strength,
                            'trend_direction': trend_direction,
                            'volatility': volatility,
                            'adx': adx,
                            'is_suitable': None  # وضعیت مناسب بودن
                        })
            
            mt5.shutdown()
            return forex_pairs
            
        except Exception as e:
            logging.error(f"Error getting symbols: {e}")
            return []

    @staticmethod
    def select_symbol():
        """انتخاب جفت ارز توسط کاربر"""
        while True:
            try:
                print(f"\n{Fore.CYAN}{'='*60}")
                print(f"{Fore.YELLOW}🌟 AVAILABLE FOREX PAIRS{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*60}\n")
                
                # لیست جفت ارزهای اصلی
                major_pairs = [
                    'EURUSDb', 'GBPUSDb', 'USDJPYb', 'USDCHFb', 
                    'AUDUSDb', 'USDCADb', 'NZDUSDb',
                    'EURGBPb', 'EURJPYb', 'GBPJPYb'
                ]
                
                # فیلتر کردن فقط جفت ارزهای اصلی
                available_pairs = []
                symbols = mt5.symbols_get()
                
                for symbol in symbols:
                    if symbol.name in major_pairs:
                        if symbol.name not in self.suitable_pairs:
                            is_suitable, reason = self.evaluate_market_conditions(symbol.name)
                            self.suitable_pairs[symbol.name] = (is_suitable, reason)
                        available_pairs.append(symbol)
                
                # نمایش جدول جفت ارزها
                print(f"{Fore.WHITE}{'ID':<4} {'Symbol':<10} {'Status':<15} {'Condition'}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'-'*75}{Style.RESET_ALL}")
                
                for i, symbol in enumerate(available_pairs, 1):
                    is_suitable, reason = self.suitable_pairs[symbol.name]
                    status_color = Fore.GREEN if is_suitable else Fore.RED
                    status_icon = "✅" if is_suitable else "⚠️"
                    status_text = "SUITABLE" if is_suitable else "UNSUITABLE"
                    
                    print(f"{Fore.YELLOW}{i:<4}{Style.RESET_ALL}"
                          f"{Fore.WHITE}{symbol.name:<10}{Style.RESET_ALL}"
                          f"{status_color}{status_icon} {status_text:<8}{Style.RESET_ALL}"
                          f"{Fore.CYAN}{reason}{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                
                # دریافت انتخاب کاربر
                choice = input(f"\n{Fore.GREEN}📊 Enter the number of your desired pair (1-{len(available_pairs)}) or 'q' to quit: {Style.RESET_ALL}")
                
                if choice.lower() == 'q':
                    return None
                    
                index = int(choice) - 1
                if 0 <= index < len(available_pairs):
                    selected_symbol = available_pairs[index].name
                    is_suitable = self.suitable_pairs[selected_symbol][0]
                    
                    if not is_suitable:
                        print(f"\n{Fore.RED}⚠️ Warning: This pair shows unstable market conditions.{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Would you like to select a different pair? (y/n){Style.RESET_ALL}")
                        if input().lower() == 'y':
                            continue
                    print(f"\n{Fore.GREEN}✨ Selected: {selected_symbol}{Style.RESET_ALL}")
                    return selected_symbol
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                    
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}. Please try again.{Style.RESET_ALL}")
                continue

    def __init__(self, api_key, mt5_login, mt5_password, mt5_server="", symbol=None, risk_percent=2):
        """Initialize the bot with required parameters"""
        try:
            # تنظیم متغیرهای کنترل سیگنال
            self.last_signal_time = None
            self.last_signal_expiry = None
            self.current_symbol_analysis = None
            self.suitable_pairs = {}
            self.min_volume = 1000  # حذف min_adx
            self.last_signal_type = None
            
            # تنظیم Gemini با REST API
            genai.configure(api_key=api_key, transport="rest")
            self.model = genai.GenerativeModel('gemini-pro')
            
            # اگر جفت ارز مشخص نشده، از کاربر بپرس
            if symbol is None:
                print(f"\n{Fore.CYAN}Please select a currency pair...{Style.RESET_ALL}")
                symbol = self.select_symbol()
                if symbol is None:
                    raise Exception("No symbol selected")
            
            # تنظیم پارامترهای اصلی
            self.symbol = symbol
            self.risk_percent = risk_percent
            self.mt5_login = mt5_login
            self.mt5_password = mt5_password
            self.mt5_server = mt5_server
            
            print(f"\n{Fore.GREEN}Selected symbol: {self.symbol}{Style.RESET_ALL}")
            
            # تنظیم مسیر فایل صوتی
            self.sound_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1.mp3")
            
            # راه‌اندازی شبکه عصبی کمکی
            self.initialize_neural_network()
            
            # راه‌اندازی pygame mixer
            self.setup_sound_system()
            
            # تنظیم سیستم لاگینگ
            self.setup_logging()
            
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
        """تحلیل ترکیبی پرایس اکشن و هوش مصنوعی با چندین لایه اعتبارسنجی"""
        try:
            if not self.can_generate_signal():
                remaining_time = self.last_signal_expiry - \
                               (datetime.now() - self.last_signal_time).total_seconds() / 60
                print(f"\n{Fore.YELLOW}⏳ Waiting for previous signal to expire. {remaining_time:.1f} minutes remaining.{Style.RESET_ALL}")
                return None

            if data is None:
                logging.error("No market data available")
                return None

            # لایه 1: تحلیل تکنیکال پیشرفته
            technical_analysis = self.advanced_technical_analysis(data)
            if not technical_analysis['is_tradeable']:
                print(f"\n{Fore.RED}⚠️ Market conditions not suitable: {technical_analysis['reason']}{Style.RESET_ALL}")
                return None

            # لایه 2: تحلیل پرایس اکشن
            pa_analysis = self.analyze_price_action(data)
            if not self.validate_price_action(pa_analysis):
                print(f"\n{Fore.YELLOW}⚠️ Price action patterns are not clear{Style.RESET_ALL}")
                return None

            # لایه 3: تحلیل حجم و نقدینگی
            volume_analysis = self.analyze_volume_and_liquidity(data)
            if not volume_analysis['is_valid']:
                print(f"\n{Fore.YELLOW}⚠️ Volume conditions not met: {volume_analysis['reason']}{Style.RESET_ALL}")
                return None

            # لایه 4: بررسی اخبار مهم
            news_impact = self.check_news_impact()
            if news_impact:
                print(f"\n{Fore.RED}⚠️ High impact news detected - avoiding trade{Style.RESET_ALL}")
                return None

            # لایه 5: تحلیل روند بلندمدت
            trend_analysis = self.analyze_long_term_trend(data)
            
            # ساخت متن تحلیل برای AI
            market_context = f"""
            Comprehensive Market Analysis for {self.symbol}:

            Technical Analysis:
            - Current Price: {data['close'].iloc[-1]:.5f}
            - RSI: {technical_analysis['rsi']:.2f}
            - MACD: {technical_analysis['macd']:.5f}
            - Volatility: {technical_analysis['volatility']:.5f}
            
            Price Action:
            {self.format_patterns(pa_analysis['patterns'])}
            
            Volume Analysis:
            - Current Volume: {volume_analysis['current_volume']}
            - Average Volume: {volume_analysis['average_volume']}
            - Volume Trend: {volume_analysis['trend']}
            
            Long-term Trend:
            - Main Trend: {trend_analysis['main_trend']}
            - Trend Strength: {trend_analysis['strength']}
            - Key Levels: {trend_analysis['key_levels']}
            
            Market Structure:
            - Support Levels: {technical_analysis['support_levels']}
            - Resistance Levels: {technical_analysis['resistance_levels']}
            - Market Phase: {technical_analysis['market_phase']}
            """

            # لایه 6: تحلیل هوش مصنوعی با پارامترهای سختگیرانه
            prompt = f"""As a professional forex trading expert, analyze this comprehensive market data and provide a trading signal.
            
            {market_context}
            
            Strict Rules:
            1. Signal must have MULTIPLE confirmations from different indicators
            2. Volume must support the signal direction
            3. Price action must show clear patterns
            4. Signal must align with the main trend
            5. Risk/Reward ratio must be at least 1:2
            6. Minimum confidence threshold is 75%
            7. Avoid trading during high impact news
            
            Respond EXACTLY in this format:
            SIGNAL: [UP/DOWN/WAIT]
            CONFIDENCE: [75-100]
            EXPIRY: [1-5]
            ANALYSIS: [detailed explanation]
            """

            response = self.model.generate_content(prompt)
            if response is None:
                logging.error("No response from AI model")
                return None

            # لایه 7: اعتبارسنجی نهایی سیگنال
            parsed_response = self._parse_binary_response(response.text)
            if parsed_response:
                signal_type, confidence, expiry, analysis = parsed_response

                # لایه 8: تطبیق با تحلیل ماشین لرنینگ
                ml_signal = self.process_with_neural_network(data)
                if ml_signal:
                    ml_signal_type, ml_confidence = ml_signal
                    if ml_signal_type != signal_type and ml_confidence > 75:
                        print(f"\n{Fore.RED}⚠️ ML model disagrees with AI analysis{Style.RESET_ALL}")
                        return None

                # لایه 9: بررسی ریسک به ریوارد
                if not self.validate_risk_reward(signal_type, data):
                    print(f"\n{Fore.RED}⚠️ Risk/Reward ratio not favorable{Style.RESET_ALL}")
                    return None

                # لایه 10: تایید نهایی با شرایط بازار
                if not self.final_market_validation(signal_type, data):
                    print(f"\n{Fore.RED}⚠️ Final market validation failed{Style.RESET_ALL}")
                    return None

                # ذخیره زمان و مدت سیگنال
                self.last_signal_time = datetime.now()
                self.last_signal_expiry = expiry
                
                return signal_type, confidence, expiry, analysis
            else:
                logging.error("Could not parse AI response")
                return None
                
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            return None

    def advanced_technical_analysis(self, data):
        """تحلیل تکنیکال پیشرفته"""
        try:
            df = data.copy()
            last_candle = df.iloc[-1]
            
            # محاسبه شاخص‌های تکنیکال
            rsi = last_candle['RSI']
            macd = last_candle['MACD']
            volatility = df['Volatility'].iloc[-1]
            
            # تشخیص فاز بازار
            market_phase = self.determine_market_phase(df)
            
            # محاسبه سطوح کلیدی
            support_levels = self.find_key_levels(df)
            resistance_levels = [level for level in support_levels if level['type'] == 'resistance']
            support_levels = [level for level in support_levels if level['type'] == 'support']
            
            # شرایط معامله - حذف شرط ADX
            is_tradeable = True
            reason = ""
            
            if volatility > df['Volatility'].mean() * 2:
                is_tradeable = False
                reason = "Excessive volatility"
            elif market_phase == 'choppy':
                is_tradeable = False
                reason = "Choppy market conditions"
                
            return {
                'is_tradeable': is_tradeable,
                'reason': reason,
                'rsi': rsi,
                'macd': macd,
                'volatility': volatility,
                'market_phase': market_phase,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            logging.error(f"Error in advanced technical analysis: {e}")
            return {
                'is_tradeable': False,
                'reason': f"Analysis error: {str(e)}"
            }

    def determine_market_phase(self, data):
        """تشخیص فاز بازار"""
        try:
            df = data.copy()
            
            # محاسبه میانگین‌های متحرک
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # محاسبه شیب میانگین‌ها
            sma20_slope = (df['SMA20'].iloc[-1] - df['SMA20'].iloc[-10]) / 10
            sma50_slope = (df['SMA50'].iloc[-1] - df['SMA50'].iloc[-10]) / 10
            
            # محاسبه نوسانات
            volatility = df['close'].pct_change().std()
            avg_volatility = df['close'].pct_change().std().mean()
            
            if abs(sma20_slope) < 0.0001 and abs(sma50_slope) < 0.0001:
                return 'ranging'
            elif volatility > avg_volatility * 2:
                return 'choppy'
            elif sma20_slope > 0 and sma50_slope > 0:
                return 'uptrend'
            elif sma20_slope < 0 and sma50_slope < 0:
                return 'downtrend'
            else:
                return 'transitioning'
                
        except Exception as e:
            logging.error(f"Error determining market phase: {e}")
            return 'unknown'

    def analyze_volume_and_liquidity(self, data):
        """تحلیل حجم و نقدینگی"""
        try:
            df = data.copy()
            current_volume = df['tick_volume'].iloc[-1]
            avg_volume = df['tick_volume'].rolling(window=20).mean().iloc[-1]
            
            # محاسبه روند حجم
            volume_trend = 'increasing' if df['tick_volume'].iloc[-3:].is_monotonic_increasing else 'decreasing'
            
            # بررسی شرایط حجم
            is_valid = True
            reason = ""
            
            if current_volume < avg_volume * 0.7:
                is_valid = False
                reason = "Low volume"
            elif current_volume > avg_volume * 3:
                is_valid = False
                reason = "Suspicious volume spike"
                
            return {
                'is_valid': is_valid,
                'reason': reason,
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'trend': volume_trend
            }
            
        except Exception as e:
            logging.error(f"Error in volume analysis: {e}")
            return {
                'is_valid': False,
                'reason': f"Analysis error: {str(e)}",
                'current_volume': 0,
                'average_volume': 0,
                'trend': 'unknown'
            }

    def analyze_long_term_trend(self, data):
        """تحلیل روند بلندمدت"""
        try:
            df = data.copy()
            
            # محاسبه میانگین‌های متحرک بلندمدت
            df['SMA100'] = df['close'].rolling(window=100).mean()
            df['SMA200'] = df['close'].rolling(window=200).mean()
            
            current_price = df['close'].iloc[-1]
            sma100 = df['SMA100'].iloc[-1]
            sma200 = df['SMA200'].iloc[-1]
            
            # تشخیص روند اصلی
            if current_price > sma100 and current_price > sma200:
                main_trend = 'bullish'
            elif current_price < sma100 and current_price < sma200:
                main_trend = 'bearish'
            else:
                main_trend = 'neutral'
            
            # محاسبه قدرت روند
            trend_strength = self.calculate_trend_strength(df)
            
            # یافتن سطوح کلیدی
            key_levels = self.find_key_levels(df)
            
            return {
                'main_trend': main_trend,
                'strength': trend_strength,
                'key_levels': key_levels
            }
            
        except Exception as e:
            logging.error(f"Error in long-term trend analysis: {e}")
            return {
                'main_trend': 'unknown',
                'strength': 0,
                'key_levels': []
            }

    def calculate_trend_strength(self, data):
        """محاسبه قدرت روند"""
        try:
            df = data.copy()
            
            # محاسبه ADX
            adx = self.calculate_adx(df).iloc[-1]
            
            # محاسبه شیب قیمت
            price_slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / 20
            
            # محاسبه نسبت کندل‌های هم‌جهت با روند
            trend_candles = 0
            total_candles = min(20, len(df))
            
            for i in range(1, total_candles):
                if price_slope > 0 and df['close'].iloc[-i] > df['open'].iloc[-i]:
                    trend_candles += 1
                elif price_slope < 0 and df['close'].iloc[-i] < df['open'].iloc[-i]:
                    trend_candles += 1
            
            trend_quality = trend_candles / total_candles
            
            # ترکیب فاکتورها
            strength = (adx / 100 + trend_quality) / 2
            
            return min(max(strength, 0), 1)  # نرمال‌سازی بین 0 و 1
            
        except Exception as e:
            logging.error(f"Error calculating trend strength: {e}")
            return 0

    def validate_risk_reward(self, signal_type, data):
        """بررسی نسبت ریسک به ریوارد"""
        try:
            df = data.copy()
            current_price = df['close'].iloc[-1]
            
            # یافتن نزدیک‌ترین سطوح حمایت و مقاومت
            levels = self.find_key_levels(df)
            supports = [l['price'] for l in levels if l['type'] == 'support']
            resistances = [l['price'] for l in levels if l['type'] == 'resistance']
            
            if signal_type == 'UP':
                # برای سیگنال خرید
                stop_loss = min(supports) if supports else (current_price - (current_price * 0.001))
                take_profit = min(resistances) if resistances else (current_price + (current_price * 0.002))
                
                risk = current_price - stop_loss
                reward = take_profit - current_price
                
            else:  # signal_type == 'DOWN'
                # برای سیگنال فروش
                stop_loss = max(resistances) if resistances else (current_price + (current_price * 0.001))
                take_profit = max(supports) if supports else (current_price - (current_price * 0.002))
                
                risk = stop_loss - current_price
                reward = current_price - take_profit
            
            # محاسبه نسبت
            if risk == 0:
                return False
                
            rr_ratio = abs(reward / risk)
            return rr_ratio >= 2  # حداقل نسبت 1:2
            
        except Exception as e:
            logging.error(f"Error validating risk/reward: {e}")
            return False

    def final_market_validation(self, signal_type, data):
        """اعتبارسنجی نهایی شرایط بازار"""
        try:
            df = data.copy()
            
            # بررسی واگرایی‌ها
            divergence = self.check_divergences(df)
            if divergence['exists'] and divergence['type'] != signal_type:
                return False
            
            # بررسی الگوهای هارمونیک
            harmonic = self.check_harmonic_patterns(df)
            if harmonic['exists'] and harmonic['type'] != signal_type:
                return False
            
            # بررسی شکست‌های قیمتی
            breakout = self.check_breakouts(df)
            if breakout['exists'] and breakout['type'] != signal_type:
                return False
            
            # بررسی حجم معاملات
            volume_trend = self.analyze_volume_and_liquidity(df)
            if not volume_trend['is_valid']:
                return False
            
            # بررسی نوسانات غیرعادی
            volatility = df['Volatility'].iloc[-1]
            avg_volatility = df['Volatility'].mean()
            if volatility > avg_volatility * 2:
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in final market validation: {e}")
            return False

    def check_divergences(self, data):
        """بررسی واگرایی‌های قیمت و اندیکاتورها"""
        try:
            df = data.copy()
            
            # بررسی واگرایی RSI
            price_higher = df['close'].iloc[-1] > df['close'].iloc[-2]
            rsi_higher = df['RSI'].iloc[-1] > df['RSI'].iloc[-2]
            
            if price_higher != rsi_higher:
                return {
                    'exists': True,
                    'type': 'DOWN' if price_higher else 'UP',
                    'indicator': 'RSI'
                }
            
            # بررسی واگرایی MACD
            macd_higher = df['MACD'].iloc[-1] > df['MACD'].iloc[-2]
            
            if price_higher != macd_higher:
                return {
                    'exists': True,
                    'type': 'DOWN' if price_higher else 'UP',
                    'indicator': 'MACD'
                }
            
            return {
                'exists': False,
                'type': None,
                'indicator': None
            }
            
        except Exception as e:
            logging.error(f"Error checking divergences: {e}")
            return {
                'exists': False,
                'type': None,
                'indicator': None
            }

    def check_harmonic_patterns(self, data):
        """بررسی الگوهای هارمونیک"""
        try:
            df = data.copy()
            
            # یافتن نقاط سوئینگ
            swings = self.find_swing_points(df)
            
            # بررسی الگوی گارتلی
            gartley = self.check_gartley_pattern(swings)
            if gartley['exists']:
                return gartley
            
            # بررسی الگوی پروانه
            butterfly = self.check_butterfly_pattern(swings)
            if butterfly['exists']:
                return butterfly
            
            # بررسی سایر الگوهای هارمونیک
            # ...
            
            return {
                'exists': False,
                'type': None,
                'pattern': None
            }
            
        except Exception as e:
            logging.error(f"Error checking harmonic patterns: {e}")
            return {
                'exists': False,
                'type': None,
                'pattern': None
            }

    def check_breakouts(self, data):
        """بررسی شکست‌های قیمتی"""
        try:
            df = data.copy()
            current_price = df['close'].iloc[-1]
            
            # یافتن سطوح کلیدی
            levels = self.find_key_levels(df)
            
            # بررسی شکست سطوح
            for level in levels:
                # شکست سطح مقاومت
                if level['type'] == 'resistance' and current_price > level['price']:
                    return {
                        'exists': True,
                        'type': 'UP',
                        'level': level['price']
                    }
                # شکست سطح حمایت
                elif level['type'] == 'support' and current_price < level['price']:
                    return {
                        'exists': True,
                        'type': 'DOWN',
                        'level': level['price']
                    }
            
            return {
                'exists': False,
                'type': None,
                'level': None
            }
            
        except Exception as e:
            logging.error(f"Error checking breakouts: {e}")
            return {
                'exists': False,
                'type': None,
                'level': None
            }

    def find_swing_points(self, data):
        """یافتن نقاط سوئینگ قیمت"""
        try:
            df = data.copy()
            swings = []
            
            for i in range(2, len(df) - 2):
                # سوئینگ بالا
                if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                    df['high'].iloc[i] > df['high'].iloc[i-2] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    swings.append({
                        'type': 'high',
                        'price': df['high'].iloc[i],
                        'index': i
                    })
                
                # سوئینگ پایین
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['low'].iloc[i] < df['low'].iloc[i-2] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    swings.append({
                        'type': 'low',
                        'price': df['low'].iloc[i],
                        'index': i
                    })
            
            return swings
            
        except Exception as e:
            logging.error(f"Error finding swing points: {e}")
            return []

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
            if not response:
                logging.warning("Empty response from AI")
                return None
                
            signal_type = None
            confidence = 0
            expiry = 0
            analysis = ""
            
            # لاگ کردن پاسخ خام برای دیباگ
            logging.debug(f"Raw AI response:\n{response}")
            
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                try:
                    if line.upper().startswith('SIGNAL:'):
                        signal_text = line.split(':', 1)[1].strip().upper()
                        if 'N/A' in signal_text:
                            signal_type = 'WAIT'
                        elif 'UP' in signal_text:
                            signal_type = 'UP'
                        elif 'DOWN' in signal_text:
                            signal_type = 'DOWN'
                        elif 'WAIT' in signal_text:
                            signal_type = 'WAIT'
                        logging.debug(f"Parsed signal type: {signal_type}")
                        
                    elif line.upper().startswith('CONFIDENCE:'):
                        confidence_text = line.split(':', 1)[1].strip()
                        if 'N/A' in confidence_text:
                            confidence = 0
                        else:
                            # حذف کاراکترهای غیر عددی به جز نقطه اعشار
                            confidence_clean = ''.join(c for c in confidence_text if c.isdigit() or c == '.')
                            if confidence_clean:
                                confidence = float(confidence_clean)
                        logging.debug(f"Parsed confidence: {confidence}")
                        
                    elif line.upper().startswith('EXPIRY:'):
                        expiry_text = line.split(':', 1)[1].strip()
                        if 'N/A' in expiry_text:
                            expiry = 1
                        else:
                            # حذف همه کاراکترها به جز اعداد
                            expiry_clean = ''.join(filter(str.isdigit, expiry_text))
                            if expiry_clean:
                                expiry = int(expiry_clean)
                            else:
                                expiry = 1
                        logging.debug(f"Parsed expiry: {expiry}")
                        
                    elif line.upper().startswith('ANALYSIS:'):
                        analysis = line.split(':', 1)[1].strip()
                        if analysis.upper() == 'N/A':
                            analysis = "No detailed analysis available"
                        logging.debug(f"Parsed analysis: {analysis}")
                        
                except Exception as e:
                    logging.warning(f"Error parsing line '{line}': {e}")
                    continue
            
            # اعتبارسنجی مقادیر
            if not signal_type:
                logging.warning("No valid signal type found")
                return None
                
            if not (0 <= confidence <= 100):
                logging.warning(f"Invalid confidence value: {confidence}")
                confidence = 0
            
            if not (1 <= expiry <= 5):
                logging.warning(f"Invalid expiry value: {expiry}")
                expiry = 1
            
            if not analysis:
                logging.warning("No analysis text found")
                analysis = "No analysis provided"
            
            # اگر اعتماد صفر است، سیگنال را به WAIT تغییر بده
            if confidence == 0:
                signal_type = 'WAIT'
                
            logging.info(f"Successfully parsed response: {signal_type}, {confidence}%, {expiry}min")
            return signal_type, confidence, expiry, analysis
            
        except Exception as e:
            logging.error(f"Error parsing AI response: {e}")
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
        
        # پاک کردن صفحه
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n{Fore.CYAN}🤖 AI FOREX SIGNAL GENERATOR STARTED{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Analyzing {self.symbol} after each 1-minute candle close{Style.RESET_ALL}\n")
        
        while True:
            try:
                current_time = datetime.now()
                current_minute = current_time.minute
                
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
                            # پخش صدای هشدار
                            self.play_alert_sound()
                            
                            # ایجاد پیام سیگنال
                            signal_message = f"""
╔══════════════════════════════════════════╗
║ {Fore.YELLOW}🔔 FOREX SIGNAL{Style.RESET_ALL}                          ║
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

    def play_alert_sound(self):
        """پخش صدای هشدار"""
        try:
            if os.path.exists(self.sound_file):
                # بازنشانی و بارگذاری مجدد فایل صوتی
                mixer.music.stop()
                mixer.music.unload()
                mixer.music.load(self.sound_file)
                mixer.music.set_volume(1.0)
                mixer.music.play()
                time.sleep(0.5)  # اجازه بده صدا پخش شود
            else:
                winsound.Beep(1000, 1000)
        except Exception as e:
            print(f"Error playing sound: {e}")
            try:
                winsound.Beep(1000, 1000)
            except:
                pass

    def evaluate_market_conditions(self, symbol):
        try:
            # Get historical data
            candles = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
            if candles is None or len(candles) < 100:
                return False, "Insufficient historical data"
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calculate ADX
            df['ADX'] = self.calculate_adx(df)
            current_adx = df['ADX'].iloc[-1]
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            current_rsi = df['RSI'].iloc[-1]
            
            # Calculate average volume
            avg_volume = df['tick_volume'].mean()
            
            # Market evaluation criteria
            conditions = []
            
            # Check ADX (trend strength)
            if current_adx >= self.min_adx:
                conditions.append(True)
            else:
                return False, f"Weak trend (ADX: {current_adx:.2f})"
            
            # Check volume
            if avg_volume >= self.min_volume:
                conditions.append(True)
            else:
                return False, f"Low volume (Avg Volume: {avg_volume:.2f})"
            
            # Check RSI (avoid overbought/oversold)
            if 30 <= current_rsi <= 70:
                conditions.append(True)
            else:
                return False, f"Extreme RSI value (RSI: {current_rsi:.2f})"
            
            # All conditions must be met
            is_suitable = all(conditions)
            reason = "Market conditions are favorable" if is_suitable else "Market conditions are unfavorable"
            
            return is_suitable, reason
            
        except Exception as e:
            print(f"Error evaluating market conditions: {str(e)}")
            return False, f"Error: {str(e)}"

    def calculate_price_stability(self, df):
        """محاسبه ثبات قیمت"""
        try:
            # محاسبه درصد تغییرات قیمت
            price_changes = abs(df['close'].pct_change())
            stability = 100 - (price_changes.mean() * 1000)  # تبدیل به درصد
            return max(0, min(100, stability))
        except:
            return 0

    def initialize_neural_network(self):
        """راه‌اندازی مدل یادگیری ماشین برای پردازش سیگنال‌ها"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # تنظیم مدل
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            
            print(f"{Fore.GREEN}Machine learning model initialized successfully{Style.RESET_ALL}")
            
        except Exception as e:
            logging.error(f"Error initializing ML model: {e}")
            self.ml_model = None
            self.scaler = None

    def process_with_neural_network(self, data):
        """پردازش داده‌ها با مدل یادگیری ماشین"""
        try:
            if self.ml_model is None or self.scaler is None:
                return None
                
            # آماده‌سازی داده‌ها
            features = ['close', 'open', 'high', 'low', 'RSI', 'MACD', 'Volatility']
            X = data[features].values
            
            # نرمال‌سازی داده‌ها
            X = self.scaler.fit_transform(X)
            
            # میانگین‌گیری از ویژگی‌ها برای یک پیش‌بینی
            X_mean = X.mean(axis=0).reshape(1, -1)
            
            # پیش‌بینی
            # اینجا فقط احتمالات را برمی‌گرداند چون مدل هنوز آموزش ندیده است
            probabilities = np.array([0.33, 0.33, 0.34])  # مقادیر پیش‌فرض تا زمان آموزش مدل
            
            # تبدیل خروجی به سیگنال
            signal_map = {0: 'UP', 1: 'DOWN', 2: 'WAIT'}
            signal = signal_map[np.argmax(probabilities)]
            confidence = float(np.max(probabilities)) * 100
            
            return signal, confidence
            
        except Exception as e:
            logging.error(f"Error in ML model processing: {e}")
            return None

    def is_active_trading_hours(self):
        """بررسی ساعات فعال معاملاتی"""
        current_time = datetime.now()
        # ساعات فعال: دوشنبه تا جمعه، 00:05 تا 23:55
        return current_time.weekday() < 5 and \
               (current_time.hour > 0 or (current_time.hour == 0 and current_time.minute >= 5)) and \
               (current_time.hour < 23 or (current_time.hour == 23 and current_time.minute <= 55))

    def can_generate_signal(self):
        """بررسی امکان تولید سیگنال جدید"""
        if self.last_signal_time is None or self.last_signal_expiry is None:
            return True
            
        time_passed = (datetime.now() - self.last_signal_time).total_seconds() / 60
        return time_passed >= self.last_signal_expiry

    def select_symbol(self):
        """انتخاب جفت ارز توسط کاربر"""
        while True:
            try:
                print(f"\n{Fore.CYAN}{'='*60}")
                print(f"{Fore.YELLOW}🌟 AVAILABLE FOREX PAIRS{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*60}\n")
                
                # لیست جفت ارزهای اصلی
                major_pairs = [
                    'EURUSDb', 'GBPUSDb', 'USDJPYb', 'USDCHFb', 
                    'AUDUSDb', 'USDCADb', 'NZDUSDb',
                    'EURGBPb', 'EURJPYb', 'GBPJPYb'
                ]
                
                # فیلتر کردن فقط جفت ارزهای اصلی
                available_pairs = []
                symbols = mt5.symbols_get()
                
                for symbol in symbols:
                    if symbol.name in major_pairs:
                        if symbol.name not in self.suitable_pairs:
                            is_suitable, reason = self.evaluate_market_conditions(symbol.name)
                            self.suitable_pairs[symbol.name] = (is_suitable, reason)
                        available_pairs.append(symbol)
                
                # نمایش جدول جفت ارزها
                print(f"{Fore.WHITE}{'ID':<4} {'Symbol':<10} {'Status':<15} {'Condition'}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'-'*75}{Style.RESET_ALL}")
                
                for i, symbol in enumerate(available_pairs, 1):
                    is_suitable, reason = self.suitable_pairs[symbol.name]
                    status_color = Fore.GREEN if is_suitable else Fore.RED
                    status_icon = "✅" if is_suitable else "⚠️"
                    status_text = "SUITABLE" if is_suitable else "UNSUITABLE"
                    
                    print(f"{Fore.YELLOW}{i:<4}{Style.RESET_ALL}"
                          f"{Fore.WHITE}{symbol.name:<10}{Style.RESET_ALL}"
                          f"{status_color}{status_icon} {status_text:<8}{Style.RESET_ALL}"
                          f"{Fore.CYAN}{reason}{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                
                # دریافت انتخاب کاربر
                choice = input(f"\n{Fore.GREEN}📊 Enter the number of your desired pair (1-{len(available_pairs)}) or 'q' to quit: {Style.RESET_ALL}")
                
                if choice.lower() == 'q':
                    return None
                    
                index = int(choice) - 1
                if 0 <= index < len(available_pairs):
                    selected_symbol = available_pairs[index].name
                    is_suitable = self.suitable_pairs[selected_symbol][0]
                    
                    if not is_suitable:
                        print(f"\n{Fore.RED}⚠️ Warning: This pair shows unstable market conditions.{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}Would you like to select a different pair? (y/n){Style.RESET_ALL}")
                        if input().lower() == 'y':
                            continue
                    print(f"\n{Fore.GREEN}✨ Selected: {selected_symbol}{Style.RESET_ALL}")
                    return selected_symbol
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                    
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error: {e}. Please try again.{Style.RESET_ALL}")
                continue

    @staticmethod
    def calculate_adx(df, period=14):
        """محاسبه ADX به صورت دستی"""
        try:
            df = df.copy()
            df['TR'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['DMplus'] = np.where(
                (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                np.maximum(df['high'] - df['high'].shift(1), 0),
                0
            )
            df['DMminus'] = np.where(
                (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                np.maximum(df['low'].shift(1) - df['low'], 0),
                0
            )
            
            df['TRn'] = df['TR'].rolling(window=period).mean()
            df['DMplusN'] = df['DMplus'].rolling(window=period).mean()
            df['DMminusN'] = df['DMminus'].rolling(window=period).mean()
            
            df['DIplus'] = 100 * df['DMplusN'] / df['TRn']
            df['DIminus'] = 100 * df['DMminusN'] / df['TRn']
            
            df['DIdiff'] = abs(df['DIplus'] - df['DIminus'])
            df['DIsum'] = df['DIplus'] + df['DIminus']
            
            df['DX'] = 100 * df['DIdiff'] / df['DIsum']
            df['ADX'] = df['DX'].rolling(window=period).mean()
            
            return df['ADX']
            
        except Exception as e:
            logging.error(f"Error calculating ADX: {e}")
            return pd.Series([25] * len(df))  # مقدار پیش‌فرض در صورت خطا

    def setup_sound_system(self):
        """راه‌اندازی سیستم صوتی برای هشدارها"""
        try:
            mixer.init()
            mixer.music.set_volume(1.0)
            print(f"{Fore.GREEN}✓ Sound system initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}✗ Error initializing sound system: {e}{Style.RESET_ALL}")
            logging.error(f"Sound system initialization failed: {e}")
            # در صورت خطا، از بیپ سیستمی استفاده می‌کنیم
            self.sound_file = None

    def setup_logging(self):
        """راه‌اندازی سیستم لاگینگ"""
        try:
            # تنظیم فرمت لاگ‌ها
            log_format = '%(asctime)s - %(levelname)s - %(message)s'
            
            # ایجاد مسیر برای فایل لاگ
            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # نام فایل لاگ با تاریخ
            log_file = os.path.join(log_dir, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
            
            # تنظیم لاگر
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            logging.info("Logging system initialized successfully")
            print(f"{Fore.GREEN}✓ Logging system initialized{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error initializing logging system: {e}{Style.RESET_ALL}")
            # در صورت خطا، تنظیم حداقلی لاگر
            logging.basicConfig(level=logging.INFO)
            logging.error(f"Logging system initialization failed: {e}")

    def handle_trading_panel(self, expiry_minutes):
        """Handle trading panel interactions after signal"""
        try:
            if not self.pfinance_session:
                logging.error("No active p.finance session")
                return False
                
            wait = WebDriverWait(self.pfinance_session, 10)
            
            # Click on control-buttons__wrapper
            control_buttons = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "control-buttons__wrapper")))
            control_buttons.click()
            time.sleep(1)
            
            # Click on control__value value value--several-items
            value_selector = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "control__value.value.value--several-items")))
            value_selector.click()
            time.sleep(1)
            
            # Find the trading panel modal
            trading_panel = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "trading-panel-modal__in")))
            
            # Find all rw divs
            rw_divs = trading_panel.find_elements(By.CLASS_NAME, "rw")
            
            if len(rw_divs) >= 2:  # Make sure we have at least 2 divs
                # Get the second rw div
                second_rw = rw_divs[1]
                
                # Find the input element in the second div
                input_field = second_rw.find_element(By.TAG_NAME, "input")
                
                # Clear existing value and set new expiry time
                input_field.clear()
                input_field.send_keys(str(expiry_minutes))
                time.sleep(0.5)
                
                return True
                
        except Exception as e:
            logging.error(f"Error handling trading panel: {e}")
            return False

    def login_to_pfinance(self):
        """ورود به حساب p.finance"""
        try:
            # تنظیمات مرورگر
            options = webdriver.ChromeOptions()
            options.add_argument('--start-maximized')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            # راه‌اندازی مرورگر
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            
            # باز کردن صفحه لاگین
            print(f"\n{Fore.CYAN}Opening p.finance login page...{Style.RESET_ALL}")
            driver.get("https://p.finance/en/cabinet/demo-quick-high-low/")
            
            # انتظار برای لود شدن صفحه
            wait = WebDriverWait(driver, 10)
            
            # پر کردن فرم لاگین
            print(f"{Fore.CYAN}Filling login form...{Style.RESET_ALL}")
            email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
            email_field.send_keys("mousavifarsamaneh@gmail.com")
            
            password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
            password_field.send_keys("Ms3950171533")
            
            # کلیک روی دکمه لاگین
            print(f"{Fore.CYAN}Clicking login button...{Style.RESET_ALL}")
            login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
            login_button.click()
            
            print(f"{Fore.GREEN}✓ Login successful!{Style.RESET_ALL}")
            
            # انتظار برای لود شدن صفحه بعد از لاگین
            time.sleep(3)  # صبر برای لود کامل صفحه
            
            # کلیک روی right-block__item
            print(f"{Fore.CYAN}Clicking on balance menu...{Style.RESET_ALL}")
            right_block = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.right-block__item.js-drop-down-modal-open")))
            right_block.click()
            
            # کلیک روی balance-item
            print(f"{Fore.CYAN}Selecting balance option...{Style.RESET_ALL}")
            balance_item = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.balance-item")))
            balance_item.click()
            
            print(f"{Fore.GREEN}✓ Navigation successful!{Style.RESET_ALL}")
            
            # کلیک روی pair-number-wrap
            print(f"{Fore.CYAN}Opening currency pair selection...{Style.RESET_ALL}")
            pair_selector = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.pair-number-wrap")))
            pair_selector.click()
            
            # آماده‌سازی جفت ارز برای جستجو
            search_pair = self.symbol.replace('b', '')  # حذف b از انتهای نام جفت ارز
            formatted_pair = f"{search_pair[:3]}/{search_pair[3:]}"  # اضافه کردن / بین ارزها
            
            # جستجوی جفت ارز
            print(f"{Fore.CYAN}Searching for {formatted_pair}...{Style.RESET_ALL}")
            search_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.search__field")))
            search_field.clear()
            search_field.send_keys(search_pair)
            
            time.sleep(2)  # صبر برای نمایش نتایج جستجو
            
            # انتخاب جفت ارز از لیست با مسیر درست
            print(f"{Fore.CYAN}Selecting currency pair...{Style.RESET_ALL}")
            try:
                # جستجو برای لینک حاوی جفت ارز مورد نظر
                pair_link = wait.until(EC.element_to_be_clickable((
                    By.XPATH,
                    f"//a[contains(@class, 'alist__link')]//span[contains(@class, 'alist__label') and text()='{formatted_pair}']"
                )))
                # کلیک روی لینک
                pair_link.click()
                print(f"{Fore.GREEN}✓ Currency pair {formatted_pair} selected!{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error selecting currency pair: {str(e)}{Style.RESET_ALL}")
                raise
            
            time.sleep(1)  # تاخیر کوتاه بعد از انتخاب
            
            # کلیک روی control-buttons__wrapper
            print(f"{Fore.CYAN}Setting up trading panel...{Style.RESET_ALL}")
            control_buttons = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "control-buttons__wrapper")))
            control_buttons.click()
            
            # ذخیره session برای استفاده‌های بعدی
            self.pfinance_session = driver
            print(f"{Fore.GREEN}✓ Trading panel initialized!{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error logging into p.finance: {str(e)}{Style.RESET_ALL}")
            if 'driver' in locals():
                driver.quit()
            self.pfinance_session = None

    def validate_price_action(self, pa_analysis):
        """اعتبارسنجی الگوهای پرایس اکشن"""
        try:
            if not pa_analysis:
                return False
                
            # بررسی وجود حداقل یک الگوی معتبر
            valid_patterns = 0
            
            # بررسی الگوهای کندل استیک
            for pattern_name, pattern_data in pa_analysis['patterns'].items():
                if pattern_data and pattern_data['strength'] > 0.6:  # الگوهای با قدرت بالای 60%
                    valid_patterns += 1
            
            # بررسی شکست‌های قیمتی
            for breakout in pa_analysis.get('breakout_analysis', []):
                if breakout.get('exists', False):
                    valid_patterns += 1
            
            # بررسی سطوح کلیدی
            key_levels = pa_analysis.get('key_levels', [])
            if len(key_levels) > 0:
                for level in key_levels:
                    if level.get('strength', {}).get('reliability', 0) > 0.7:  # سطوح با اعتبار بالای 70%
                        valid_patterns += 1
            
            # بررسی حجم
            volume_analysis = pa_analysis.get('volume_analysis', {})
            if volume_analysis.get('volume_surge', False):
                valid_patterns += 1
            elif volume_analysis.get('volume_trend') == 'increasing':
                valid_patterns += 0.5
            
            # نیاز به حداقل 2 تایید برای اعتبارسنجی
            return valid_patterns >= 2
            
        except Exception as e:
            logging.error(f"Error validating price action: {e}")
            return False

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

