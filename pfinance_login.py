from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def login_to_pfinance():
    try:
        # تنظیمات مرورگر
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        # راه‌اندازی مرورگر
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # باز کردن صفحه لاگین
        print("Opening p.finance login page...")
        driver.get("https://p.finance/en/cabinet/demo-quick-high-low/")
        
        # انتظار برای لود شدن صفحه
        wait = WebDriverWait(driver, 10)
        
        # پر کردن فرم لاگین
        print("Filling login form...")
        email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        email_field.send_keys("mousavifarsamaneh@gmail.com")
        
        password_field = wait.until(EC.presence_of_element_located((By.NAME, "password")))
        password_field.send_keys("Ms3950171533")
        
        # کلیک روی دکمه لاگین
        print("Clicking login button...")
        login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        login_button.click()
        
        print("Login successful!")
        
        # نگه داشتن مرورگر باز
        input("Press Enter to close the browser...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    login_to_pfinance() 