import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin
import re

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Content-Type': 'application/json',
    }

def simulate_login(username, password):
    print('\n开始模拟登录...')
    
    # 登录API端点
    login_api = 'https://account.usr.cn/api/user/login'
    
    # 准备登录数据
    login_data = {
        'username': username,
        'password': password,
        'type': 'mp_scada'
    }
    
    try:
        # 发送登录请求
        session = requests.Session()
        response = session.post(
            login_api,
            headers=get_headers(),
            json=login_data,
            timeout=10
        )
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        if result.get('code') == 0:
            print('登录成功！')
            print('用户信息:', json.dumps(result.get('data', {}), ensure_ascii=False, indent=2))
            return session, result.get('data', {})
        else:
            print('登录失败:', result.get('msg', '未知错误'))
            return None, None
            
    except requests.RequestException as e:
        print(f'登录请求失败: {str(e)}')
        return None, None
    except Exception as e:
        print(f'登录过程出错: {str(e)}')
        return None, None

def analyze_login_page(login_soup):
    # 分析登录页面结构
    print('\n分析登录页面结构...')
    
    # 查找登录相关的脚本文件
    scripts = login_soup.find_all('script')
    login_scripts = [script['src'] for script in scripts if script.get('src') and 'usrPass' in script['src']]
    if login_scripts:
        print('找到登录相关脚本:', login_scripts)
    
    # 查找可能的登录表单元素
    app_div = login_soup.find('div', id='app')
    if app_div:
        print('找到应用主容器 #app')
    
    # 分析加载状态元素
    loading_div = login_soup.find('div', id='__whole-loading')
    if loading_div:
        print('找到加载状态元素 #__whole-loading')
        loading_img = loading_div.find('img')
        if loading_img:
            print('加载图片路径:', loading_img.get('src'))
    
    # 提取所有外部资源文件
    css_files = [link['href'] for link in login_soup.find_all('link', rel='stylesheet')]
    print('\n样式文件:')
    for css in css_files:
        print(f'- {css}')
    
    js_files = [script['src'] for script in scripts if script.get('src')]
    print('\nJavaScript文件:')
    for js in js_files:
        print(f'- {js}')

def print_full_login_page_content(login_soup, login_response):
    print('\n' + '='*50)
    print('登录页面完整内容分析')
    print('='*50)
    
    # 输出基本页面信息
    print('\n1. 基本信息：')
    print(f'页面大小: {len(login_response.text)} 字节')
    print(f'页面编码: {login_response.encoding}')
    
    # 输出HTML结构
    print('\n2. HTML结构：')
    print('-'*30)
    print(login_soup.prettify())
    
    # 输出所有脚本内容
    print('\n3. 脚本文件列表：')
    print('-'*30)
    scripts = login_soup.find_all('script')
    for i, script in enumerate(scripts, 1):
        if script.get('src'):
            print(f'\n[Script {i}] 外部脚本: {script["src"]}')
        elif script.string:
            print(f'\n[Script {i}] 内联脚本:')
            print(script.string.strip())
    
    # 输出所有样式表内容
    print('\n4. 样式表列表：')
    print('-'*30)
    styles = login_soup.find_all(['style', 'link'])
    for i, style in enumerate(styles, 1):
        if style.name == 'link' and style.get('rel') == ['stylesheet']:
            print(f'\n[Style {i}] 外部样式表: {style["href"]}')
        elif style.name == 'style' and style.string:
            print(f'\n[Style {i}] 内联样式:')
            print(style.string.strip())
    
    # 输出所有图片资源
    print('\n5. 图片资源列表：')
    print('-'*30)
    images = login_soup.find_all('img')
    for i, img in enumerate(images, 1):
        print(f'[Image {i}] {img.get("src", "无源")} (alt: {img.get("alt", "无描述")})')
    
    print('\n' + '='*50)

def crawl_initial_site():
    try:
        # 访问初始网站
        initial_url = 'http://cloud.usr.cn/'
        response = requests.get(initial_url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        print(f'成功访问初始网站: {initial_url}')
        print(f'状态码: {response.status_code}')
        
        # 解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        print('初始网站HTML内容长度:', len(response.text))
        
        # 获取登录按钮相关信息
        login_url = 'https://account.usr.cn/#/login?type=mp_scada'
        print(f'登录页面URL: {login_url}')
        
        # 访问登录页面
        login_response = requests.get(login_url, headers=get_headers(), timeout=10)
        login_response.raise_for_status()
        print('\n成功访问登录页面')
        print(f'状态码: {login_response.status_code}')
        
        # 解析登录页面内容
        login_soup = BeautifulSoup(login_response.text, 'html.parser')
        print('登录页面HTML内容长度:', len(login_response.text))
        
        # 分析登录页面结构
        analyze_login_page(login_soup)
        
        # 输出登录页面的完整内容
        print_full_login_page_content(login_soup, login_response)
        
        # 保存页面内容到文件（用于调试）
        with open('initial_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        with open('login_page.html', 'w', encoding='utf-8') as f:
            f.write(login_response.text)
            
    except requests.RequestException as e:
        print(f'请求出错: {str(e)}')
    except Exception as e:
        print(f'发生错误: {str(e)}')

def crawl_homepage(session):
    if not session:
        print('未获取到有效会话，无法访问主页')
        return
    
    try:
        # 访问云平台主页
        homepage_url = 'https://mp.usr.cn/#/cloud/homePage/ViewHomePage'
        response = session.get(homepage_url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        print(f'\n成功访问云平台主页: {homepage_url}')
        print(f'状态码: {response.status_code}')
        
        # 解析主页内容
        soup = BeautifulSoup(response.text, 'html.parser')
        print('主页HTML内容长度:', len(response.text))
        
        # 分析主页结构
        print('\n分析主页结构...')
        
        # 查找主要内容容器
        main_container = soup.find('div', class_='home-page')
        if main_container:
            print('找到主页内容容器')
            
            # 提取页面标题
            title_elem = main_container.find('h1')
            if title_elem:
                print('页面标题:', title_elem.text.strip())
            
            # 提取设备信息
            device_info = main_container.find('div', class_='device-info')
            if device_info:
                print('\n设备信息:')
                print(device_info.text.strip())
            
            # 提取数据卡片
            data_cards = main_container.find_all('div', class_='data-card')
            print(f'\n找到 {len(data_cards)} 个数据卡片')
            for card in data_cards:
                card_title = card.find('div', class_='card-title')
                card_value = card.find('div', class_='card-value')
                if card_title and card_value:
                    print(f'{card_title.text.strip()}: {card_value.text.strip()}')
        
        # 保存主页内容到文件（用于调试）
        with open('homepage.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
            
    except requests.RequestException as e:
        print(f'请求主页失败: {str(e)}')
    except Exception as e:
        print(f'解析主页时发生错误: {str(e)}')

if __name__ == '__main__':
    print('开始解析云平台主页...')
    try:
        # 访问云平台主页
        homepage_url = 'https://mp.usr.cn/#/cloud/homePage/ViewHomePage'
        response = requests.get(homepage_url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        print(f'\n成功访问云平台主页: {homepage_url}')
        print(f'状态码: {response.status_code}')
        
        # 解析主页内容
        soup = BeautifulSoup(response.text, 'html.parser')
        print('主页HTML内容长度:', len(response.text))
        
        # 分析主页结构
        print('\n分析主页结构...')
        
        # 查找主要内容容器
        main_container = soup.find('div', class_='home-page')
        if main_container:
            print('找到主页内容容器')
            
            # 提取页面标题
            title_elem = main_container.find('h1')
            if title_elem:
                print('页面标题:', title_elem.text.strip())
            
            # 提取设备信息
            device_info = main_container.find('div', class_='device-info')
            if device_info:
                print('\n设备信息:')
                print(device_info.text.strip())
            
            # 提取数据卡片
            data_cards = main_container.find_all('div', class_='data-card')
            print(f'\n找到 {len(data_cards)} 个数据卡片')
            for card in data_cards:
                card_title = card.find('div', class_='card-title')
                card_value = card.find('div', class_='card-value')
                if card_title and card_value:
                    print(f'{card_title.text.strip()}: {card_value.text.strip()}')
        
        # 保存主页内容到文件（用于调试）
        with open('homepage.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
            print('\n已保存主页内容到 homepage.html')
            
    except requests.RequestException as e:
        print(f'请求主页失败: {str(e)}')
    except Exception as e:
        print(f'解析主页时发生错误: {str(e)}')