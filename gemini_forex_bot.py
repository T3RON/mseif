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
from termcolor import colored
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tradingview_ta import TA_Handler, Interval, Exchange

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
            self.min_adx = 25
            self.min_volume = 1000
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

    def can_generate_signal(self):
        """بررسی امکان تولید سیگنال جدید"""
        try:
            if not hasattr(self, 'last_signal_time') or not hasattr(self, 'last_signal_expiry'):
                return True
                
            if self.last_signal_time is None:
                return True
                
            # محاسبه زمان باقی‌مانده از سیگنال قبلی
            elapsed_time = (datetime.now() - self.last_signal_time).total_seconds() / 60
            remaining_time = self.last_signal_expiry - elapsed_time
            
            if remaining_time > 0:
                # نمایش پیام انتظار
                print(f"\n{Fore.YELLOW}⏳ Waiting for previous signal to expire...")
                print(f"   Remaining time: {remaining_time:.1f} minutes")
                print(f"   Previous signal: {self.last_signal_type} @ {self.last_signal_entry_price:.5f}")
                print(f"   Confidence: {self.last_signal_confidence}%{Style.RESET_ALL}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error checking signal generation: {e}")
            return False

    def wait_for_candle_close(self):
        """انتظار برای بسته شدن کندل جاری"""
        try:
            current_time = datetime.now()
            seconds_until_close = 60 - current_time.second  # زمان باقی‌مانده تا پایان دقیقه
            
            if seconds_until_close > 0:
                print(f"\n{Fore.YELLOW}⏳ Waiting for current candle to close...{Style.RESET_ALL}")
                
                # شمارنده معکوس با به‌روزرسانی هر ثانیه
                for remaining in range(seconds_until_close, 0, -1):
                    # پاک کردن خط قبلی
                    sys.stdout.write('\r')
                    
                    # نمایش زمان باقی‌مانده با فرمت مناسب
                    minutes = remaining // 60
                    seconds = remaining % 60
                    
                    if minutes > 0:
                        time_str = f"{minutes}m {seconds}s"
                    else:
                        time_str = f"{seconds}s"
                    
                    # نمایش نوار پیشرفت
                    progress = int((seconds_until_close - remaining) / seconds_until_close * 20)
                    progress_bar = f"[{'='*progress}{' '*(20-progress)}]"
                    
                    sys.stdout.write(f"{Fore.CYAN}{progress_bar} {Fore.YELLOW}Time remaining: {time_str}{Style.RESET_ALL}")
                    sys.stdout.flush()
                    
                    time.sleep(1)
                
                # پاک کردن خط آخر و نمایش پیام اتمام
                sys.stdout.write('\r')
                sys.stdout.write(f"{Fore.GREEN}✓ Candle closed, analyzing...{' '*50}{Style.RESET_ALL}\n")
                sys.stdout.flush()
                
                return True
                
            return True
            
        except Exception as e:
            logging.error(f"Error waiting for candle close: {e}")
            return False

    def analyze_with_gemini(self, data):
        """تحلیل ترکیبی با استفاده از داده‌های TradingView و هوش مصنوعی"""
        try:
            # بررسی امکان تولید سیگنال جدید
            if not self.can_generate_signal():
                return None
                
            # انتظار برای بسته شدن کندل جاری
            if not self.wait_for_candle_close():
                print(f"{Fore.RED}Error waiting for candle close{Style.RESET_ALL}")
                return None
                
            # به‌روزرسانی داده‌ها پس از بسته شدن کندل
            data = self.get_market_data()  # دریافت داده‌های جدید
            
            if data is None or len(data) < 2:  # حداقل به دو کندل نیاز داریم
                logging.error("Insufficient market data")
                return None
                
            # بررسی آخرین کندل
            last_candle = data.iloc[-1]
            prev_candle = data.iloc[-2]
            
            # اطمینان از کامل بودن کندل
            if abs((datetime.now() - pd.to_datetime(last_candle.name)).total_seconds()) < 60:
                logging.warning("Current candle is not yet complete")
                return None
                
            # دریافت تحلیل‌های TradingView
            tv_analysis = self.get_tradingview_analysis()
            
            # دریافت داده‌های هیت‌مپ
            tv_data = self.get_tradingview_data()
            
            # دریافت رویدادهای اقتصادی
            economic_events = self.get_economic_calendar()
            
            # بررسی رویدادهای مهم
            if economic_events:
                current_pair = self.symbol.replace('b', '')
                base = current_pair[:3]
                quote = current_pair[3:]
                
                important_events = [e for e in economic_events if e['currency'] in [base, quote] and e['impact'] == 'HIGH']
                if important_events:
                    print(f"\n{Fore.RED}⚠️ Warning: High impact events detected. Trading may be risky.{Style.RESET_ALL}")
                    return None

            if data is None:
                logging.error("No market data available")
                return None

            # آماده‌سازی داده‌های تکنیکال
            last_candle = data.iloc[-1]
            current_price = last_candle['close']
            
            # تحلیل پرایس اکشن
            pa_analysis = self.analyze_price_action(data)
            
            # اضافه کردن داده‌های TradingView به متن تحلیل
            market_context = f"""
Please analyze this forex market data and provide a detailed technical analysis with trading signal.

Current Market Data for {self.symbol}:
1. Price Action:
   - Current Price: {current_price:.5f}
   - Previous Close: {data['close'].iloc[-2]:.5f}
   - Daily Range: {(data['high'].iloc[-1] - data['low'].iloc[-1]):.5f}

2. Technical Indicators:
   - RSI: {last_candle['RSI']:.2f}
   - MACD: {last_candle['MACD']:.5f}
   - Signal Line: {last_candle['Signal_Line']:.5f}
   - 20 SMA: {last_candle['SMA20']:.5f}
   - Volatility: {last_candle['Volatility']:.5f}

3. TradingView Analysis:
   - Overall Recommendation: {tv_analysis['summary']['RECOMMENDATION'] if tv_analysis else 'N/A'}
   - Buy Signals: {tv_analysis['summary']['BUY'] if tv_analysis else 'N/A'}
   - Sell Signals: {tv_analysis['summary']['SELL'] if tv_analysis else 'N/A'}
   - RSI: {tv_analysis['indicators']['RSI'] if tv_analysis else 'N/A'}
   - ADX: {tv_analysis['indicators']['ADX'] if tv_analysis else 'N/A'}
   - CCI: {tv_analysis['indicators']['CCI'] if tv_analysis else 'N/A'}

4. Price Patterns:
   {self.format_patterns(pa_analysis['patterns'])}

5. Key Levels:
   {self.format_key_levels(pa_analysis['key_levels'])}

6. Market Heatmap:
   {self.format_tradingview_data(tv_data) if tv_data else "No heatmap data available"}

Requirements:
1. Consider TradingView's technical analysis and recommendations
2. Analyze trend strength and direction
3. Evaluate momentum using multiple indicators
4. Consider price patterns and key levels
5. Analyze cross-market correlations
6. Provide clear trading signal (UP/DOWN/WAIT)
7. Suggest optimal expiry time (5-15 minutes) - Note: Minimum expiry time is 5 minutes
8. Include confidence level (0-100%)

Please format your response exactly as follows:
SIGNAL: [UP/DOWN/WAIT]
CONFIDENCE: [0-100]
EXPIRY: [5-15]
ANALYSIS: Provide detailed technical analysis including:
- TradingView recommendations analysis
- Current trend analysis
- Support/Resistance levels
- Technical indicator readings
- Pattern formations
- Cross-market correlations
- Final recommendation with reasoning
"""

            # تحلیل FVG
            fvg_data = self.analyze_fvg(data)
            fvg_signals = self.get_fvg_signals(fvg_data, current_price)
            
            # اضافه کردن تحلیل FVG به متن تحلیل
            market_context += f"""
            
Fair Value Gap Analysis:
{self.format_fvg_analysis(fvg_data)}

FVG Signals:
{len(fvg_signals)} active signals detected
"""
            
            # ترکیب سیگنال‌های FVG با سایر تحلیل‌ها
            if fvg_signals:
                for signal in fvg_signals:
                    if signal['direction'] == signal_type:
                        confidence = min(100, confidence + signal['strength'] * 0.2)  # افزایش اطمینان
                        analysis += f"\n{signal['description']}"
            
            # تحلیل فیبوناچی
            fib_analysis = self.analyze_fibonacci(data)
            
            # اضافه کردن تحلیل فیبوناچی به متن تحلیل
            market_context += f"""

Fibonacci Analysis:
{self.format_fibonacci_analysis(fib_analysis)}

"""
            
            # ترکیب سیگنال‌های فیبوناچی با سایر تحلیل‌ها
            if fib_analysis and fib_analysis['signals']:
                for signal in fib_analysis['signals']:
                    if signal['type'] == signal_type:
                        confidence = min(100, confidence + signal['strength'] * 0.15)
                        analysis += f"\n{signal['description']}"
            
            response = self.model.generate_content(market_context)
            if response is None:
                logging.error("No response from AI model")
                return None
            
            # نمایش تحلیل با فرمت زیبا
            parsed_response = self._parse_binary_response(response.text)
            
            if parsed_response:
                signal_type, confidence, expiry, analysis = parsed_response
                
                if signal_type != "WAIT" and confidence > 60:
                    # ذخیره اطلاعات سیگنال
                    self.last_signal_time = datetime.now()
                    self.last_signal_expiry = expiry
                    self.last_signal_entry_price = current_price
                    self.last_signal_type = signal_type
                    self.last_signal_confidence = confidence
                    
                    # تنظیم تایمر برای ارزیابی نتیجه
                    timer = threading.Timer(
                        expiry * 60,
                        self.evaluate_signal_result,
                        args=[signal_type, current_price, confidence, expiry]
                    )
                    timer.start()
                
                return signal_type, confidence, expiry, analysis
            else:
                logging.error("Could not parse AI response")
                return None
                
        except Exception as e:
            logging.error(f"Error in analysis: {e}")
            return None

    def format_volume_analysis(self, volume_data):
        """فرمت‌بندی تحلیل حجم معاملات"""
        if not volume_data:
            return "No volume data available"
            
        volume_status = "High" if volume_data.get('volume_surge', False) else "Normal"
        trend = volume_data.get('volume_trend', 'neutral').capitalize()
        
        return f"Volume Status: {volume_status}, Trend: {trend}"

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
            
            # تقسیم پاسخ به خطوط و حذف خطوط خالی
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # اگر پاسخ خالی است یا فرمت درستی ندارد
            if not lines:
                logging.warning("Response contains no valid lines")
                return None

            # پردازش خط به خط پاسخ
            found_analysis = False
            analysis_lines = []
            
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
                            expiry = 5  # حداقل زمان انقضا
                        else:
                            # حذف همه کاراکترها به جز اعداد
                            expiry_clean = ''.join(filter(str.isdigit, expiry_text))
                            if expiry_clean:
                                # اعمال محدودیت حداقل 5 دقیقه
                                expiry = max(5, int(expiry_clean))
                            else:
                                expiry = 5
                        logging.debug(f"Parsed expiry: {expiry}")
                        
                    elif line.upper().startswith('ANALYSIS:'):
                        found_analysis = True
                        # اضافه کردن متن بعد از ANALYSIS: به لیست
                        initial_analysis = line.split(':', 1)[1].strip()
                        if initial_analysis and initial_analysis.upper() != 'N/A':
                            analysis_lines.append(initial_analysis)
                        
                    elif found_analysis:
                        # اضافه کردن خطوط بعدی به تحلیل
                        analysis_lines.append(line.strip())
                        
                except Exception as e:
                    logging.warning(f"Error parsing line '{line}': {e}")
                    continue
            
            # ترکیب خطوط تحلیل
            if analysis_lines:
                analysis = ' '.join(analysis_lines)
            else:
                analysis = "Market conditions are currently being analyzed. Please wait for detailed analysis."
                logging.debug("No analysis lines found, using default message")

            # اعتبارسنجی مقادیر
            if not signal_type:
                logging.warning("No valid signal type found")
                return None
                
            if not (0 <= confidence <= 100):
                logging.warning(f"Invalid confidence value: {confidence}")
                confidence = 0
            
            if not (5 <= expiry <= 15):
                logging.warning(f"Invalid expiry value: {expiry}")
                expiry = 5
            
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
        """اجرای ربات"""
        try:
            print(f"\n{Fore.GREEN}Starting trading bot...{Style.RESET_ALL}")
            
            while True:
                try:
                    if not self.is_active_trading_hours():
                        print(f"\n{Fore.YELLOW}Outside trading hours. Waiting...{Style.RESET_ALL}")
                        time.sleep(60)
                        continue
                    
                    # دریافت داده‌های بازار
                    market_data = self.get_market_data()
                    if market_data is None:
                        continue
                    
                    # انتظار برای بسته شدن کندل جاری
                    current_time = datetime.now()
                    if current_time.second != 0:  # اگر در ابتدای دقیقه نیستیم
                        self.wait_for_candle_close()
                        continue
                    
                    # تحلیل و تولید سیگنال
                    signal = self.analyze_with_gemini(market_data)
                    
                    if signal:
                        signal_type, confidence, expiry, analysis = signal
                        self.display_signal(signal_type, confidence, expiry, analysis)
                    
                    # تاخیر تا کندل بعدی
                    time.sleep(5)  # تاخیر کوتاه برای جلوگیری از فشار به CPU
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Bot stopped by user{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Fatal error: {e}")
            print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")

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

    def display_header(self):
        """نمایش هدر برنامه"""
        header = f"""
{Fore.CYAN}╔{'═'*50}╗
║{Fore.YELLOW}             🤖 GEMINI FOREX AI BOT{Fore.CYAN}              ║
║{Fore.YELLOW}          Powered by Advanced AI Analysis{Fore.CYAN}        ║
╚{'═'*50}╝{Style.RESET_ALL}
"""
        print(header)

    def display_status_bar(self, current_time, symbol, mode="analyzing"):
        """نمایش نوار وضعیت"""
        if mode == "analyzing":
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[int(time.time() * 10) % 10]
            status = f"{Fore.CYAN}{spinner} Analyzing {symbol} at {current_time.strftime('%H:%M:%S')}{Style.RESET_ALL}"
        else:
            status = f"{Fore.YELLOW}⏳ Waiting for next candle...{Style.RESET_ALL}"
        
        print(f"\r{status}", end="")

    def display_signal(self, signal_type, confidence, expiry, analysis):
        """نمایش سیگنال معاملاتی با فرمت زیبا"""
        try:
            current_time = datetime.now()
            
            # نمایش کادر سیگنال
            print(f"\n{Fore.CYAN}╔{'═'*60}╗")
            print(f"║{' '*24}TRADING SIGNAL{' '*24}║")
            print(f"╠{'═'*60}╣")
            
            # نمایش زمان و نوع سیگنال
            signal_color = Fore.GREEN if signal_type == "UP" else Fore.RED if signal_type == "DOWN" else Fore.YELLOW
            signal_icon = "🔼" if signal_type == "UP" else "🔽" if signal_type == "DOWN" else "⏸️"
            
            print(f"║ Time: {Fore.WHITE}{current_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}{' '*27}║")
            print(f"║ Signal: {signal_color}{signal_icon} {signal_type}{Style.RESET_ALL}{' '*(45-len(signal_type))}║")
            
            # نمایش اطمینان
            confidence_bar = self.get_confidence_bar(confidence)
            print(f"║ Confidence: {confidence_bar} {confidence}%{' '*(35-len(str(confidence)))}║")
            
            # نمایش زمان انقضا
            expiry_time = current_time + timedelta(minutes=expiry)
            print(f"║ Expiry: {Fore.WHITE}{expiry_time.strftime('%H:%M:%S')} ({expiry} min){Style.RESET_ALL}{' '*27}║")
            
            print(f"╠{'═'*60}╣")
            
            # نمایش تحلیل
            if analysis:
                formatted_analysis = self.format_analysis(analysis)
                print(formatted_analysis)
            
            print(f"╚{'═'*60}╝")
            
            # پخش صدای اعلان
            self.play_alert_sound()
            
        except Exception as e:
            logging.error(f"Error displaying signal: {e}")
            print(f"\n{Fore.RED}Error displaying signal: {str(e)}{Style.RESET_ALL}")

    def format_technical_analysis(self, analysis):
        """فرمت‌بندی تحلیل تکنیکال با جزئیات بیشتر"""
        # جداسازی بخش‌های مختلف تحلیل
        sections = {
            'Trend': [],
            'Support/Resistance': [],
            'Indicators': [],
            'Pattern': [],
            'Recommendation': []
        }
        
        # تقسیم متن تحلیل به خطوط
        lines = analysis.split('.')
        
        # دسته‌بندی هر خط در بخش مربوطه
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if any(word in line.lower() for word in ['trend', 'moving', 'direction']):
                sections['Trend'].append(line)
            elif any(word in line.lower() for word in ['support', 'resistance', 'level']):
                sections['Support/Resistance'].append(line)
            elif any(word in line.lower() for word in ['rsi', 'macd', 'indicator', 'oscillator']):
                sections['Indicators'].append(line)
            elif any(word in line.lower() for word in ['pattern', 'formation', 'candlestick']):
                sections['Pattern'].append(line)
            else:
                sections['Recommendation'].append(line)

        # ساخت خروجی فرمت شده
        formatted_output = []
        
        # نمایش هر بخش با آیکون مخصوص
        icons = {
            'Trend': '📈',
            'Support/Resistance': '🎯',
            'Indicators': '📊',
            'Pattern': '🔄',
            'Recommendation': '💡'
        }
        
        for section, items in sections.items():
            if items:
                formatted_output.append(f"║ {Fore.YELLOW}{icons[section]} {section}:{Style.RESET_ALL}")
                for item in items:
                    # شکستن خطوط طولانی
                    words = item.split()
                    current_line = f"║ {Fore.WHITE}  •"
                    
                    for word in words:
                        if len(current_line + " " + word) > 55:
                            formatted_output.append(current_line + " " * (54 - len(current_line)) + f"{Fore.CYAN}║")
                            current_line = f"║ {Fore.WHITE}    "
                        current_line += " " + word
                    
                    if current_line:
                        formatted_output.append(current_line + " " * (54 - len(current_line)) + f"{Fore.CYAN}║")
                formatted_output.append(f"║{' ' * 58}{Fore.CYAN}║")
        
        return "\n".join(formatted_output)

    def get_confidence_bar(self, confidence):
        """ایجاد نوار گرافیکی برای نمایش اطمینان"""
        bar_length = 20
        filled_length = int(confidence / 100 * bar_length)
        bar = ''
        
        if confidence >= 75:
            bar = f"{Fore.GREEN}"
        elif confidence >= 60:
            bar = f"{Fore.YELLOW}"
        else:
            bar = f"{Fore.RED}"
            
        bar += "█" * filled_length
        bar += f"{Fore.WHITE}▒" * (bar_length - filled_length)
        
        return bar

    def format_analysis(self, analysis):
        """فرمت‌بندی متن تحلیل"""
        formatted_lines = []
        words = analysis.split()
        current_line = f"║ {Fore.WHITE}"
        
        for word in words:
            if len(current_line + word) > 55:  # حداکثر طول خط
                formatted_lines.append(current_line + " " * (54 - len(current_line)) + f"{Fore.CYAN}║")
                current_line = f"║ {Fore.WHITE}{word}"
            else:
                current_line += " " + word
        
        if current_line:
            formatted_lines.append(current_line + " " * (54 - len(current_line)) + f"{Fore.CYAN}║")
            
        return "\n".join(formatted_lines)

    def get_market_trend(self, data):
        """تشخیص روند کلی بازار"""
        try:
            sma20 = data['SMA20'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            if current_price > sma20:
                return f"{Fore.GREEN}Bullish ↗{Style.RESET_ALL}"
            else:
                return f"{Fore.RED}Bearish ↘{Style.RESET_ALL}"
        except:
            return f"{Fore.YELLOW}Neutral →{Style.RESET_ALL}"

    def calculate_risk_level(self, data):
        """محاسبه سطح ریسک معامله"""
        try:
            volatility = data['Volatility'].iloc[-1]
            avg_volatility = data['Volatility'].mean()
            
            if volatility > avg_volatility * 1.5:
                return f"{Fore.RED}High{Style.RESET_ALL}"
            elif volatility > avg_volatility * 1.2:
                return f"{Fore.YELLOW}Medium{Style.RESET_ALL}"
            else:
                return f"{Fore.GREEN}Low{Style.RESET_ALL}"
        except:
            return f"{Fore.YELLOW}Unknown{Style.RESET_ALL}"

    def get_rhythmic_data(self):
        """دریافت داده‌های ریتمیک از MetaTrader5"""
        try:
            while True:
                # دریافت تیک‌های قیمت
                ticks = mt5.copy_ticks_from(self.symbol, datetime.now(), 1000, mt5.COPY_TICKS_ALL)
                if ticks is None:
                    continue
                    
                df_ticks = pd.DataFrame(ticks)
                df_ticks['time'] = pd.to_datetime(df_ticks['time'], unit='s')
                
                # محاسبه شاخص‌های لحظه‌ای
                current_tick = df_ticks.iloc[-1]
                bid = current_tick['bid']
                ask = current_tick['ask']
                volume = current_tick['volume']
                
                # نمایش اطلاعات لحظه‌ای
                self.display_rhythmic_data(bid, ask, volume)
                
                # تاخیر کوتاه
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Rhythmic data monitoring stopped.{Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Error in rhythmic data: {e}")

    def display_rhythmic_data(self, bid, ask, volume):
        """نمایش داده‌های ریتمیک"""
        spread = (ask - bid) * 10000  # محاسبه اسپرد به پیپ
        
        print(f"\r{Fore.CYAN}Bid: {Fore.WHITE}{bid:.5f} {Fore.CYAN}| "
              f"Ask: {Fore.WHITE}{ask:.5f} {Fore.CYAN}| "
              f"Spread: {Fore.WHITE}{spread:.1f} {Fore.CYAN}pips | "
              f"Volume: {Fore.WHITE}{volume}{Style.RESET_ALL}", end="")

    def start_rhythmic_monitoring(self):
        """شروع مانیتورینگ داده‌های ریتمیک"""
        print(f"\n{Fore.CYAN}Starting rhythmic data monitoring for {self.symbol}...{Style.RESET_ALL}")
        
        # ایجاد یک thread جداگانه برای مانیتورینگ داده‌های ریتمیک
        rhythmic_thread = threading.Thread(target=self.get_rhythmic_data)
        rhythmic_thread.daemon = True  # thread به محض بسته شدن برنامه متوقف شود
        rhythmic_thread.start()
        
        return rhythmic_thread

    def get_tradingview_data(self):
        """دریافت داده‌های بازار از TradingView"""
        try:
            # تنظیم هدرهای درخواست
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            # درخواست به API هیت‌مپ TradingView
            url = "https://www.tradingview.com/markets/currencies/cross-rates-overview-heat-map/"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # پردازش داده‌های هیت‌مپ
                market_data = {
                    'cross_rates': {},
                    'market_sentiment': {},
                    'currency_strength': {}
                }
                
                # استخراج نرخ‌های متقاطع
                for row in soup.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        currency_pair = cells[0].text.strip()
                        rate = cells[1].text.strip()
                        market_data['cross_rates'][currency_pair] = rate
                
                logging.info(f"Successfully retrieved TradingView data for {len(market_data['cross_rates'])} currency pairs")
                return market_data
                
            else:
                logging.error(f"Failed to get TradingView data: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting TradingView data: {e}")
            return None

    def format_tradingview_data(self, tv_data):
        """فرمت‌بندی داده‌های TradingView برای تحلیل"""
        if not tv_data:
            return "No TradingView data available"
            
        formatted_data = []
        
        # فرمت‌بندی نرخ‌های متقاطع
        if tv_data.get('cross_rates'):
            formatted_data.append("Cross Rates:")
            for pair, rate in tv_data['cross_rates'].items():
                formatted_data.append(f"- {pair}: {rate}")
        
        # فرمت‌بندی سایر داده‌ها
        if tv_data.get('market_sentiment'):
            formatted_data.append("\nMarket Sentiment:")
            for currency, sentiment in tv_data['market_sentiment'].items():
                formatted_data.append(f"- {currency}: {sentiment}")
                
        return '\n'.join(formatted_data)

    def display_heatmap_summary(self, tv_data):
        """نمایش خلاصه هیت‌مپ بازار"""
        try:
            if not tv_data or not tv_data.get('cross_rates'):
                print(f"\n{Fore.YELLOW}⚠️ No heatmap data available{Style.RESET_ALL}")
                return

            print(f"\n{Fore.CYAN}{'='*50}")
            print(f"{Fore.YELLOW}📊 FOREX HEATMAP SUMMARY{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}\n")

            # تحلیل قدرت ارزها
            currency_strength = {
                'USD': 0, 'EUR': 0, 'GBP': 0, 'JPY': 0,
                'AUD': 0, 'NZD': 0, 'CAD': 0, 'CHF': 0
            }

            # محاسبه قدرت هر ارز
            for pair, rate in tv_data['cross_rates'].items():
                try:
                    base = pair[:3]
                    quote = pair[3:]
                    change = float(rate.replace('%', ''))
                    
                    if base in currency_strength:
                        currency_strength[base] += change
                    if quote in currency_strength:
                        currency_strength[quote] -= change
                except:
                    continue

            # نمایش قدرت‌ترین و ضعیف‌ترین ارزها
            sorted_currencies = sorted(currency_strength.items(), key=lambda x: x[1], reverse=True)
            
            print(f"{Fore.WHITE}Strongest Currencies:{Style.RESET_ALL}")
            for currency, strength in sorted_currencies[:3]:
                color = Fore.GREEN if strength > 0 else Fore.RED
                print(f"  {color}{currency}: {strength:+.2f}%{Style.RESET_ALL}")

            print(f"\n{Fore.WHITE}Weakest Currencies:{Style.RESET_ALL}")
            for currency, strength in sorted_currencies[-3:]:
                color = Fore.GREEN if strength > 0 else Fore.RED
                print(f"  {color}{currency}: {strength:+.2f}%{Style.RESET_ALL}")

            # نمایش وضعیت جفت ارز فعلی
            current_pair = self.symbol.replace('b', '')
            base = current_pair[:3]
            quote = current_pair[3:]
            
            base_strength = currency_strength.get(base, 0)
            quote_strength = currency_strength.get(quote, 0)
            relative_strength = base_strength - quote_strength
            
            print(f"\n{Fore.WHITE}Current Pair Analysis ({current_pair}):{Style.RESET_ALL}")
            print(f"  {base}: {Fore.GREEN if base_strength > 0 else Fore.RED}{base_strength:+.2f}%{Style.RESET_ALL}")
            print(f"  {quote}: {Fore.GREEN if quote_strength > 0 else Fore.RED}{quote_strength:+.2f}%{Style.RESET_ALL}")
            
            bias = "Bullish" if relative_strength > 0 else "Bearish"
            bias_color = Fore.GREEN if relative_strength > 0 else Fore.RED
            print(f"  Bias: {bias_color}{bias}{Style.RESET_ALL} (Relative Strength: {relative_strength:+.2f}%)")
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")

        except Exception as e:
            logging.error(f"Error displaying heatmap summary: {e}")
            print(f"\n{Fore.RED}Error displaying heatmap summary{Style.RESET_ALL}")

    def get_tradingview_analysis(self):
        """دریافت تحلیل‌های تکنیکال از TradingView"""
        try:
            # تبدیل نام جفت ارز به فرمت مورد نیاز TradingView
            symbol = self.symbol.replace('b', '')
            
            # راه‌اندازی هندلر TradingView
            handler = TA_Handler(
                symbol=symbol,
                screener="forex",
                exchange="FX_IDC",
                interval=Interval.INTERVAL_1_MINUTE
            )
            
            # دریافت تحلیل‌ها
            analysis = handler.get_analysis()
            
            if analysis:
                try:
                    # ساختار داده برای ذخیره تحلیل‌ها با مدیریت خطا برای هر فیلد
                    tv_analysis = {
                        'summary': {
                            'RECOMMENDATION': analysis.summary.get('RECOMMENDATION', 'N/A'),
                            'BUY': analysis.summary.get('BUY', 0),
                            'SELL': analysis.summary.get('SELL', 0),
                            'NEUTRAL': analysis.summary.get('NEUTRAL', 0)
                        },
                        'oscillators': {
                            'RECOMMENDATION': analysis.oscillators.get('RECOMMENDATION', 'N/A'),
                            'BUY': analysis.oscillators.get('BUY', 0),
                            'SELL': analysis.oscillators.get('SELL', 0),
                            'NEUTRAL': analysis.oscillators.get('NEUTRAL', 0)
                        },
                        'moving_averages': {
                            'RECOMMENDATION': analysis.moving_averages.get('RECOMMENDATION', 'N/A'),
                            'BUY': analysis.moving_averages.get('BUY', 0),
                            'SELL': analysis.moving_averages.get('SELL', 0),
                            'NEUTRAL': analysis.moving_averages.get('NEUTRAL', 0)
                        },
                        'indicators': {}
                    }
                    
                    # اضافه کردن شاخص‌ها با بررسی وجود هر کدام
                    indicator_keys = [
                        'RSI', 'Stoch.K', 'CCI', 'ADX', 'AO', 'Mom', 
                        'MACD.macd', 'Stoch.RSI.K', 'W.R', 'BBPower', 'UO'
                    ]
                    
                    for key in indicator_keys:
                        try:
                            value = analysis.indicators.get(key, 0)
                            # تبدیل به float با مدیریت خطا
                            tv_analysis['indicators'][key] = float(value) if value is not None else 0
                        except (ValueError, TypeError):
                            tv_analysis['indicators'][key] = 0
                            logging.warning(f"Could not convert indicator {key} to float")
                    
                    # نمایش خلاصه تحلیل
                    self.display_tradingview_analysis(tv_analysis)
                    
                    return tv_analysis
                    
                except Exception as e:
                    logging.error(f"Error processing TradingView analysis data: {e}")
                    return None
                
            return None
            
        except Exception as e:
            logging.error(f"Error getting TradingView analysis: {e}")
            return None

    def display_tradingview_analysis(self, analysis):
        """نمایش تحلیل‌های TradingView"""
        try:
            print(f"\n{Fore.CYAN}{'='*50}")
            print(f"{Fore.YELLOW}📊 TRADINGVIEW TECHNICAL ANALYSIS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*50}\n")
            
            # نمایش توصیه کلی با مدیریت خطا
            recommendation = analysis['summary'].get('RECOMMENDATION', 'N/A')
            rec_color = (Fore.GREEN if 'BUY' in str(recommendation).upper() else 
                        Fore.RED if 'SELL' in str(recommendation).upper() else 
                        Fore.YELLOW)
            print(f"{Fore.WHITE}Overall Recommendation: {rec_color}{recommendation}{Style.RESET_ALL}")
            
            # نمایش آمار خرید/فروش
            print(f"\n{Fore.WHITE}Signal Summary:{Style.RESET_ALL}")
            print(f"  Buy Signals    : {Fore.GREEN}{analysis['summary'].get('BUY', 0)}{Style.RESET_ALL}")
            print(f"  Sell Signals   : {Fore.RED}{analysis['summary'].get('SELL', 0)}{Style.RESET_ALL}")
            print(f"  Neutral Signals: {Fore.YELLOW}{analysis['summary'].get('NEUTRAL', 0)}{Style.RESET_ALL}")
            
            # نمایش شاخص‌های مهم با بررسی وجود هر شاخص
            print(f"\n{Fore.WHITE}Key Indicators:{Style.RESET_ALL}")
            indicators = analysis.get('indicators', {})
            
            indicator_display = [
                ('RSI', 'RSI'),
                ('Stoch.K', 'Stochastic'),
                ('CCI', 'CCI'),
                ('ADX', 'ADX'),
                ('MACD.macd', 'MACD')
            ]
            
            for key, label in indicator_display:
                value = indicators.get(key, 'N/A')
                if value != 'N/A':
                    try:
                        print(f"  {label:<12}: {Fore.CYAN}{float(value):.2f}{Style.RESET_ALL}")
                    except (ValueError, TypeError):
                        print(f"  {label:<12}: {Fore.YELLOW}N/A{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
            
        except Exception as e:
            logging.error(f"Error displaying TradingView analysis: {e}")
            print(f"\n{Fore.RED}Error displaying analysis: {str(e)}{Style.RESET_ALL}")

    def evaluate_signal_result(self, signal_type, entry_price, confidence, expiry_minutes):
        """ارزیابی نتیجه سیگنال و آموزش هوش مصنوعی"""
        try:
            # دریافت قیمت فعلی
            current_tick = mt5.symbol_info_tick(self.symbol)
            if not current_tick:
                logging.error("Could not get current price")
                return
                
            exit_price = current_tick.ask if signal_type == "UP" else current_tick.bid
            
            # محاسبه درصد سود/ضرر
            if signal_type == "UP":
                profit_pips = (exit_price - entry_price) * 10000
            else:
                profit_pips = (entry_price - exit_price) * 10000
                
            # تعیین موفقیت سیگنال
            is_successful = profit_pips > 0
            
            # نمایش نتیجه
            print(f"\n{Fore.CYAN}{'='*60}")
            print(f"{Fore.YELLOW}🎯 SIGNAL RESULT ANALYSIS{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}\n")
            
            # نمایش جزئیات معامله
            print(f"{Fore.WHITE}Signal Details:{Style.RESET_ALL}")
            print(f"  Direction : {Fore.GREEN if signal_type == 'UP' else Fore.RED}{signal_type}{Style.RESET_ALL}")
            print(f"  Entry    : {Fore.CYAN}{entry_price:.5f}{Style.RESET_ALL}")
            print(f"  Exit     : {Fore.CYAN}{exit_price:.5f}{Style.RESET_ALL}")
            print(f"  Profit   : {Fore.GREEN if profit_pips > 0 else Fore.RED}{profit_pips:.1f} pips{Style.RESET_ALL}")
            
            # نمایش آمار
            result_color = Fore.GREEN if is_successful else Fore.RED
            result_text = "SUCCESSFUL" if is_successful else "FAILED"
            accuracy = abs(profit_pips / 2)  # تبدیل پیپ به درصد دقت
            
            print(f"\n{Fore.WHITE}Result:{Style.RESET_ALL}")
            print(f"  Status   : {result_color}{result_text}{Style.RESET_ALL}")
            print(f"  Accuracy : {result_color}{min(accuracy, 100):.1f}%{Style.RESET_ALL}")
            
            # آماده‌سازی داده‌ها برای آموزش AI
            training_data = {
                'signal_type': signal_type,
                'confidence': confidence,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pips': profit_pips,
                'is_successful': is_successful,
                'market_conditions': self.get_market_conditions()
            }
            
            # آموزش AI با نتیجه
            self.train_ai_with_result(training_data)
            
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            
            return is_successful, profit_pips
            
        except Exception as e:
            logging.error(f"Error evaluating signal result: {e}")
            return None, 0

    def get_market_conditions(self):
        """دریافت شرایط بازار در زمان سیگنال"""
        try:
            data = self.get_market_data()
            tv_analysis = self.get_tradingview_analysis()
            
            return {
                'technical_indicators': {
                    'RSI': data['RSI'].iloc[-1],
                    'MACD': data['MACD'].iloc[-1],
                    'SMA20': data['SMA20'].iloc[-1],
                    'Volatility': data['Volatility'].iloc[-1]
                },
                'tradingview_analysis': tv_analysis if tv_analysis else {},
                'price_action': self.analyze_price_action(data),
                'market_sentiment': self.analyze_market_sentiment()
            }
        except Exception as e:
            logging.error(f"Error getting market conditions: {e}")
            return {}

    def train_ai_with_result(self, training_data):
        """آموزش هوش مصنوعی با نتیجه سیگنال"""
        try:
            # ساخت متن آموزشی برای AI
            training_context = f"""
Analyze this forex trading signal result and provide specific feedback:

Signal Details:
- Direction: {training_data['signal_type']}
- Entry Price: {training_data['entry_price']:.5f}
- Exit Price: {training_data['exit_price']:.5f}
- Profit/Loss: {training_data['profit_pips']:.1f} pips
- Success: {'Yes' if training_data['is_successful'] else 'No'}
- Initial Confidence: {training_data['confidence']}%

Market Conditions:
1. Technical Analysis:
   - RSI: {training_data['market_conditions']['technical_indicators']['RSI']:.2f}
   - MACD: {training_data['market_conditions']['technical_indicators']['MACD']:.5f}
   - SMA20: {training_data['market_conditions']['technical_indicators']['SMA20']:.5f}
   - Volatility: {training_data['market_conditions']['technical_indicators']['Volatility']:.5f}

2. TradingView Signals:
   {self.format_tradingview_data(training_data['market_conditions']['tradingview_analysis'])}

3. Price Action Patterns:
   {self.format_patterns(training_data['market_conditions']['price_action']['patterns'])}

Based on this result, please provide:
1. A detailed analysis of why this signal succeeded or failed
2. Specific improvements for future signals in similar conditions
3. Suggested adjustments to confidence calculations

Format your response EXACTLY as follows:
ANALYSIS: [Explain why the signal succeeded/failed, focusing on key indicators and market conditions]
IMPROVEMENTS: [List specific improvements for future signals]
CONFIDENCE_ADJUSTMENTS: [Explain how to adjust confidence calculations]
"""

            # ارسال به AI برای یادگیری
            response = self.model.generate_content(training_context)
            
            if response and hasattr(response, 'text'):
                print(f"\n{Fore.MAGENTA}AI Learning Feedback:{Style.RESET_ALL}")
                feedback = self.format_ai_feedback(response.text)
                if feedback:
                    print(feedback)
                else:
                    print(f"{Fore.YELLOW}No structured feedback received from AI{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}No response received from AI{Style.RESET_ALL}")
                
        except Exception as e:
            logging.error(f"Error training AI: {e}")
            print(f"{Fore.RED}Error getting AI feedback: {str(e)}{Style.RESET_ALL}")

    def format_ai_feedback(self, feedback):
        """فرمت‌بندی بازخورد AI"""
        try:
            if not feedback:
                return None
                
            formatted = []
            current_section = None
            section_content = []
            
            for line in feedback.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('ANALYSIS:'):
                    if current_section and section_content:
                        formatted.extend(self._format_section(current_section, section_content))
                    current_section = 'ANALYSIS'
                    section_content = [line.replace('ANALYSIS:', '').strip()]
                elif line.startswith('IMPROVEMENTS:'):
                    if current_section and section_content:
                        formatted.extend(self._format_section(current_section, section_content))
                    current_section = 'IMPROVEMENTS'
                    section_content = [line.replace('IMPROVEMENTS:', '').strip()]
                elif line.startswith('CONFIDENCE_ADJUSTMENTS:'):
                    if current_section and section_content:
                        formatted.extend(self._format_section(current_section, section_content))
                    current_section = 'CONFIDENCE_ADJUSTMENTS'
                    section_content = [line.replace('CONFIDENCE_ADJUSTMENTS:', '').strip()]
                elif current_section:
                    section_content.append(line)
            
            # اضافه کردن آخرین بخش
            if current_section and section_content:
                formatted.extend(self._format_section(current_section, section_content))
            
            return '\n'.join(formatted) if formatted else None
            
        except Exception as e:
            logging.error(f"Error formatting AI feedback: {e}")
            return None

    def _format_section(self, section, content):
        """فرمت‌بندی هر بخش از بازخورد"""
        formatted = []
        if section == 'ANALYSIS':
            formatted.append(f"\n{Fore.YELLOW}📊 Analysis:{Style.RESET_ALL}")
            for line in content:
                formatted.append(f"  {line}")
        elif section == 'IMPROVEMENTS':
            formatted.append(f"\n{Fore.GREEN}📈 Improvements:{Style.RESET_ALL}")
            for i, line in enumerate(content, 1):
                formatted.append(f"  {i}. {line}")
        elif section == 'CONFIDENCE_ADJUSTMENTS':
            formatted.append(f"\n{Fore.CYAN}🎯 Confidence Adjustments:{Style.RESET_ALL}")
            for line in content:
                formatted.append(f"  • {line}")
        return formatted

    def get_economic_calendar(self):
        """دریافت و نمایش رویدادهای مهم تقویم اقتصادی"""
        try:
            # تنظیم هدرهای درخواست با User-Agent کامل‌تر
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # دریافت تاریخ امروز
            today = datetime.now().strftime('%Y-%m-%d')
            
            # درخواست به API تقویم اقتصادی
            url = f"https://www.forexfactory.com/calendar?day={today}"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                print(f"\n{Fore.CYAN}{'='*60}")
                print(f"{Fore.YELLOW}📅 ECONOMIC CALENDAR EVENTS{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*60}\n")
                
                # پردازش رویدادها
                events = []
                calendar_table = soup.find('table', class_='calendar__table')
                
                if calendar_table:
                    for row in calendar_table.find_all('tr', class_='calendar__row'):
                        try:
                            # استخراج اطلاعات رویداد
                            time_cell = row.find('td', class_='calendar__time')
                            currency = row.find('td', class_='calendar__currency')
                            impact = row.find('td', class_='calendar__impact')
                            event = row.find('td', class_='calendar__event')
                            
                            if time_cell and currency and impact and event:
                                event_time = time_cell.text.strip()
                                event_currency = currency.text.strip()
                                event_impact = self._get_impact_level(impact)
                                event_name = event.text.strip()
                                
                                events.append({
                                    'time': event_time,
                                    'currency': event_currency,
                                    'impact': event_impact,
                                    'name': event_name
                                })
                        except Exception as e:
                            logging.warning(f"Error parsing event row: {e}")
                            continue
                
                # نمایش رویدادها با فرمت مناسب
                if events:
                    print(f"{Fore.WHITE}Upcoming Events:{Style.RESET_ALL}")
                    for event in events:
                        impact_color = self._get_impact_color(event['impact'])
                        print(f"\n  {Fore.CYAN}{event['time']} {event['currency']}{Style.RESET_ALL}")
                        print(f"  {impact_color}[{event['impact']}]{Style.RESET_ALL} {event['name']}")
                else:
                    print(f"{Fore.YELLOW}No upcoming events found{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                
                # بررسی رویدادهای مهم برای جفت ارز فعلی
                current_pair = self.symbol.replace('b', '')
                base = current_pair[:3]
                quote = current_pair[3:]
                
                important_events = [e for e in events if e['currency'] in [base, quote] and e['impact'] in ['HIGH', 'MEDIUM']]
                
                if important_events:
                    print(f"\n{Fore.RED}⚠️ Warning: Important events for {current_pair}:{Style.RESET_ALL}")
                    for event in important_events:
                        impact_color = self._get_impact_color(event['impact'])
                        print(f"  • {event['time']} {impact_color}[{event['impact']}]{Style.RESET_ALL} {event['name']}")
                
                return events
                
            else:
                logging.error(f"Failed to get economic calendar: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting economic calendar: {e}")
            return None

    def _get_impact_level(self, impact_cell):
        """تعیین سطح تاثیر رویداد"""
        try:
            impact_class = impact_cell.find('span')['class'][0]
            if 'high' in impact_class:
                return 'HIGH'
            elif 'medium' in impact_class:
                return 'MEDIUM'
            elif 'low' in impact_class:
                return 'LOW'
            return 'UNKNOWN'
        except:
            return 'UNKNOWN'

    def _get_impact_color(self, impact):
        """تعیین رنگ نمایش سطح تاثیر"""
        if impact == 'HIGH':
            return Fore.RED
        elif impact == 'MEDIUM':
            return Fore.YELLOW
        elif impact == 'LOW':
            return Fore.GREEN
        return Fore.WHITE

    def analyze_fvg(self, data):
        """تحلیل شکاف‌های ارزش منصفانه (FVG)"""
        try:
            df = data.copy()
            fvg_data = {
                'bullish_fvgs': [],
                'bearish_fvgs': [],
                'active_fvgs': []
            }
            
            # بررسی 3 کندل متوالی برای FVG
            for i in range(2, len(df)-1):
                # Bullish FVG
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    fvg = {
                        'type': 'bullish',
                        'top': df['low'].iloc[i],
                        'bottom': df['high'].iloc[i-2],
                        'time': df.index[i],
                        'size': df['low'].iloc[i] - df['high'].iloc[i-2],
                        'status': 'active'
                    }
                    fvg_data['bullish_fvgs'].append(fvg)
                
                # Bearish FVG
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    fvg = {
                        'type': 'bearish',
                        'top': df['low'].iloc[i-2],
                        'bottom': df['high'].iloc[i],
                        'time': df.index[i],
                        'size': df['low'].iloc[i-2] - df['high'].iloc[i],
                        'status': 'active'
                    }
                    fvg_data['bearish_fvgs'].append(fvg)
            
            # بررسی FVG‌های فعال
            current_price = df['close'].iloc[-1]
            for fvg in fvg_data['bullish_fvgs'] + fvg_data['bearish_fvgs']:
                if fvg['bottom'] <= current_price <= fvg['top']:
                    fvg['status'] = 'active'
                    fvg_data['active_fvgs'].append(fvg)
            
            return fvg_data
            
        except Exception as e:
            logging.error(f"Error analyzing FVG: {e}")
            return None

    def get_fvg_signals(self, fvg_data, current_price):
        """استخراج سیگنال‌های معاملاتی از FVG"""
        try:
            signals = []
            
            if not fvg_data:
                return signals
                
            # بررسی FVG‌های فعال
            for fvg in fvg_data['active_fvgs']:
                # سیگنال صعودی
                if fvg['type'] == 'bullish' and current_price < fvg['bottom']:
                    signals.append({
                        'direction': 'UP',
                        'strength': min(100, fvg['size'] * 10000),  # تبدیل سایز به درصد قدرت
                        'target': fvg['top'],
                        'description': f"Bullish FVG detected: {fvg['bottom']:.5f} - {fvg['top']:.5f}"
                    })
                
                # سیگنال نزولی
                elif fvg['type'] == 'bearish' and current_price > fvg['top']:
                    signals.append({
                        'direction': 'DOWN',
                        'strength': min(100, fvg['size'] * 10000),
                        'target': fvg['bottom'],
                        'description': f"Bearish FVG detected: {fvg['top']:.5f} - {fvg['bottom']:.5f}"
                    })
            
            return signals
            
        except Exception as e:
            logging.error(f"Error getting FVG signals: {e}")
            return []

    def format_fvg_analysis(self, fvg_data):
        """فرمت‌بندی تحلیل FVG برای نمایش"""
        try:
            if not fvg_data:
                return ""
                
            output = []
            output.append(f"║ {Fore.YELLOW}🎯 Fair Value Gaps (FVG):{Style.RESET_ALL}")
            
            # نمایش FVG‌های فعال
            if fvg_data['active_fvgs']:
                output.append(f"║ {Fore.WHITE}  Active FVGs:{Style.RESET_ALL}")
                for fvg in fvg_data['active_fvgs']:
                    direction = "🔼" if fvg['type'] == 'bullish' else "🔽"
                    output.append(f"║ {Fore.WHITE}    {direction} {fvg['type'].title()}: {fvg['bottom']:.5f} - {fvg['top']:.5f}{Style.RESET_ALL}")
                    output.append(f"║ {Fore.WHITE}      Size: {fvg['size']*10000:.1f} pips{Style.RESET_ALL}")
            
            # نمایش آخرین FVG‌های تشکیل شده
            recent_bullish = [fvg for fvg in fvg_data['bullish_fvgs'][-3:]]
            recent_bearish = [fvg for fvg in fvg_data['bearish_fvgs'][-3:]]
            
            if recent_bullish or recent_bearish:
                output.append(f"║ {Fore.WHITE}  Recent FVGs:{Style.RESET_ALL}")
                for fvg in recent_bullish + recent_bearish:
                    direction = "🔼" if fvg['type'] == 'bullish' else "🔽"
                    output.append(f"║ {Fore.WHITE}    {direction} {fvg['type'].title()}: {fvg['bottom']:.5f} - {fvg['top']:.5f}{Style.RESET_ALL}")
            
            return "\n".join(output)
            
        except Exception as e:
            logging.error(f"Error formatting FVG analysis: {e}")
            return ""

    def analyze_fibonacci(self, data):
        """تحلیل سطوح فیبوناچی"""
        try:
            df = data.copy()
            
            # یافتن نقاط سوینگ (بالاترین و پایین‌ترین قیمت‌ها)
            high = df['high'].max()
            low = df['low'].min()
            current_price = df['close'].iloc[-1]
            
            # محاسبه سطوح فیبوناچی
            diff = high - low
            levels = {
                '0.0': low,
                '0.236': low + 0.236 * diff,
                '0.382': low + 0.382 * diff,
                '0.5': low + 0.5 * diff,
                '0.618': low + 0.618 * diff,
                '0.786': low + 0.786 * diff,
                '1.0': high
            }
            
            # تعیین موقعیت فعلی قیمت نسبت به سطوح فیبوناچی
            current_fib_zone = None
            next_resistance = None
            next_support = None
            
            sorted_levels = sorted(levels.items(), key=lambda x: float(x[1]))
            for i, (level, price) in enumerate(sorted_levels):
                if current_price <= price:
                    if i > 0:
                        current_fib_zone = (sorted_levels[i-1][0], level)
                        next_resistance = price
                        next_support = sorted_levels[i-1][1]
                    break
            
            # تحلیل روند با استفاده از سطوح فیبوناچی
            fib_analysis = {
                'levels': levels,
                'current_zone': current_fib_zone,
                'next_resistance': next_resistance,
                'next_support': next_support,
                'trend': self.analyze_fib_trend(current_price, levels),
                'signals': self.get_fib_signals(current_price, levels)
            }
            
            return fib_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing Fibonacci: {e}")
            return None

    def analyze_fib_trend(self, current_price, fib_levels):
        """تحلیل روند بر اساس سطوح فیبوناچی"""
        try:
            # محاسبه موقعیت نسبی قیمت در سطوح فیبوناچی
            total_range = fib_levels['1.0'] - fib_levels['0.0']
            relative_pos = (current_price - fib_levels['0.0']) / total_range
            
            if relative_pos > 0.618:
                return "Strong Bullish"
            elif relative_pos > 0.5:
                return "Bullish"
            elif relative_pos > 0.382:
                return "Neutral"
            elif relative_pos > 0.236:
                return "Bearish"
            else:
                return "Strong Bearish"
                
        except Exception as e:
            logging.error(f"Error analyzing Fibonacci trend: {e}")
            return "Unknown"

    def get_fib_signals(self, current_price, fib_levels):
        """دریافت سیگنال‌های معاملاتی بر اساس سطوح فیبوناچی"""
        try:
            signals = []
            
            # یافتن نزدیک‌ترین سطوح حمایت و مقاومت
            closest_support = None
            closest_resistance = None
            
            for level, price in fib_levels.items():
                if price < current_price and (closest_support is None or price > closest_support):
                    closest_support = price
                elif price > current_price and (closest_resistance is None or price < closest_resistance):
                    closest_resistance = price
            
            # تولید سیگنال بر اساس فاصله از سطوح
            if closest_support and closest_resistance:
                support_distance = current_price - closest_support
                resistance_distance = closest_resistance - current_price
                
                # اگر قیمت به سطح حمایت نزدیک است
                if support_distance < (resistance_distance * 0.2):
                    signals.append({
                        'type': 'UP',
                        'strength': min(100, int((1 - support_distance/resistance_distance) * 100)),
                        'description': f"Price near Fibonacci support at {closest_support:.5f}"
                    })
                
                # اگر قیمت به سطح مقاومت نزدیک است
                elif resistance_distance < (support_distance * 0.2):
                    signals.append({
                        'type': 'DOWN',
                        'strength': min(100, int((1 - resistance_distance/support_distance) * 100)),
                        'description': f"Price near Fibonacci resistance at {closest_resistance:.5f}"
                    })
            
            return signals
            
        except Exception as e:
            logging.error(f"Error getting Fibonacci signals: {e}")
            return []

    def format_fibonacci_analysis(self, fib_analysis):
        """فرمت‌بندی تحلیل فیبوناچی برای نمایش"""
        try:
            if not fib_analysis:
                return ""
                
            output = []
            output.append(f"║ {Fore.YELLOW}📐 Fibonacci Analysis:{Style.RESET_ALL}")
            
            # نمایش روند فعلی
            trend_color = {
                'Strong Bullish': Fore.GREEN,
                'Bullish': Fore.GREEN,
                'Neutral': Fore.YELLOW,
                'Bearish': Fore.RED,
                'Strong Bearish': Fore.RED
            }.get(fib_analysis['trend'], Fore.WHITE)
            
            output.append(f"║ {Fore.WHITE}  Trend: {trend_color}{fib_analysis['trend']}{Style.RESET_ALL}")
            
            # نمایش زون فعلی
            if fib_analysis['current_zone']:
                start, end = fib_analysis['current_zone']
                output.append(f"║ {Fore.WHITE}  Current Zone: {Fore.CYAN}{start} - {end}{Style.RESET_ALL}")
            
            # نمایش سطوح کلیدی
            output.append(f"║ {Fore.WHITE}  Key Levels:{Style.RESET_ALL}")
            for level, price in fib_analysis['levels'].items():
                output.append(f"║ {Fore.WHITE}    {level}: {Fore.CYAN}{price:.5f}{Style.RESET_ALL}")
            
            # نمایش سیگنال‌ها
            if fib_analysis['signals']:
                output.append(f"║ {Fore.WHITE}  Signals:{Style.RESET_ALL}")
                for signal in fib_analysis['signals']:
                    signal_color = Fore.GREEN if signal['type'] == 'UP' else Fore.RED
                    output.append(f"║ {signal_color}    • {signal['description']} ({signal['strength']}%){Style.RESET_ALL}")
            
            return "\n".join(output)
            
        except Exception as e:
            logging.error(f"Error formatting Fibonacci analysis: {e}")
            return ""

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

