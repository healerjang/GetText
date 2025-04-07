import os

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import keyboard
import random
import hashlib
from urllib.parse import urljoin, urlparse, urlunparse
import re

from socks import HTTPError

from h5py_dataset import H5pyDatafile

# 랜덤 페이지를 돌며 폰트 데이터를 수집하는 코드
# 기본 아이디어는 사이트 하나 설정 후 사이트 내에 있는 모든 링크(예: a태그로 감싸져있는 href 형태의 데이터)를 방문할 데이터 데이터셋에 저장.
# 후에 해당 데이터셋 중 랜덤하게 하나를 방문하여 스크립트 형태로 폰트 형태의 데이터를 가져와 다운로드


def compute_hash(content):
    return hashlib.sha256(content).hexdigest()

def fix_url(base_url, relative_url):
    full_url = urljoin(base_url, relative_url)
    parsed = urlparse(full_url)
    clean = parsed._replace(query="", params="", fragment="")
    return urlunparse(clean)

def get_web_link(driver, link):
    driver.get(link)
    time.sleep(2)

    links = driver.find_elements(By.TAG_NAME, "a")
    link_list = []

    for link in links:
        href = link.get_attribute("href")
        if href:
            link_list.append(href)

    font_urls = get_web_font_urls(driver)

    return font_urls, link_list

def get_web_font_urls(driver):
    js_script = """
        var urls = [];
        for (var i = 0; i < document.styleSheets.length; i++){
            try {
                var rules = document.styleSheets[i].cssRules;
                for (var j = 0; j < rules.length; j++){
                    if (rules[j].type === CSSRule.FONT_FACE_RULE) {
                        var src = rules[j].style.getPropertyValue('src');
                        urls.push(src);
                    }
                }
            } catch(e) {
                // 외부 스타일시트 등 접근이 제한된 경우 무시
            }
        }
        return urls;
        """
    font_srcs = driver.execute_script(js_script)
    font_urls = []
    pattern = re.compile(r'url\(["\']?([^)"\']+)["\']?\)')
    for src in font_srcs:
        urls = pattern.findall(src)
        font_urls.extend(urls)

    return font_urls

def font_download(font_url, save_path, font_hashes):
    response = requests.get(font_url)
    if response.status_code != 200:
        raise HTTPError(response.status_code)

    font_content = response.content
    font_hash = compute_hash(font_content)
    filename = os.path.basename(font_url)

    if font_hash not in font_hashes:
        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(font_content)

        font_hashes.add(font_hash)

def main():
    driver = webdriver.Chrome()
    url_path = "C:/SpliceImageTextData/urlPath.h5"
    font_path = "C:/SpliceImageTextData/fonts"

    # if os.path.exists(url_path):
    #     os.remove(url_path)
    dataset = H5pyDatafile(url_path, batch_size=1024)
    first_link = 'https://noonnu.cc/font_page/pick'

    visited = set()
    to_visits = {first_link}
    font_hashes = set()

    def add_set_visited(data):
        visited.add(data)
    def add_set_to_visits(data):
        visited.add(data)
    def add_set_font_hashes(data):
        visited.add(data)

    dataset.get_all('to_visit', add_set_to_visits)
    dataset.get_all('visited', add_set_visited)
    dataset.get_all('font_hashes', add_set_font_hashes)

    try:
        while True:
            if keyboard.is_pressed('f11'):
                break

            try:
                # 아직 방문하지 않은 링크 중 하나를 랜덤 선택
                to_visit = random.choice([x for x in to_visits if x not in visited])
            except IndexError:
                break
            visited.add(to_visit)
            try:
                font_urls, links = get_web_link(driver, to_visit)
            except Exception as e:
                print("링크를 가져오는 중 에러 발생:", e)
                links = []
                font_urls = []

            for link in links:
                to_visits.add(link)

            for font_url in font_urls:
                try:
                    font_download(fix_url(to_visit, font_url), font_path, font_hashes)
                except HTTPError as e:
                    pass

        driver.quit()
    except Exception as e:
        print(e)

    for to_visit in to_visits:
        dataset.save('to_visit', to_visit)

    for visit in visited:
        dataset.save('visited', visit)

    for font_hash in font_hashes:
        dataset.save('font_hashes', font_hash)

def link_test():
    path = "C:/SpliceImageTextData/urlPath.h5"
    dataset = H5pyDatafile(path, batch_size=1024)

    def __test__(i):
        print(i)

    dataset.get_all('to_visit', __test__)

if __name__ == "__main__":
    main()
