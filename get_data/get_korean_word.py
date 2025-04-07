from h5py_dataset import H5pyDatafile
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import unicodedata
from tqdm import tqdm
import random


# 한국 단어를 수집하는 코드
# 일반적인 사이트 대상 크롤링 코드

def get_urls(driver, base_url):
    driver.get(base_url)
    time.sleep(2)
    links = []

    url_table = driver.find_element(By.ID, "ENTRIES")
    tbody = url_table.find_element(By.TAG_NAME, "tbody")
    link_tds = tbody.find_elements(By.CLASS_NAME, "subject")

    for link_td in link_tds:
        a_link = link_td.find_element(By.TAG_NAME, "a")
        link = a_link.get_attribute("href")
        links.append(link)

    return links

def get_words(driver, url, dataset):
    driver.get(url)
    time.sleep(2)

    entry_content = driver.find_element(By.ID, "ENTRY-CONTENT")
    content = entry_content.find_element(By.CLASS_NAME, "tcontent")

    ps = content.find_elements(By.TAG_NAME, "p")

    for p in ps:
        text = p.text.strip()
        if text == "":
            continue

        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
        words = text.split(" ")
        for word in words:
            word = word.strip()
            if word != "":
                dataset.save('word', word)

def main():
    base_url = "https://novel.munpia.com/460563/page/1"
    driver = webdriver.Chrome()
    dataset = H5pyDatafile("C:/SpliceImageTextData/dataset/words.h5")

    word_links = get_urls(driver, base_url)
    for link in tqdm(word_links):
        get_words(driver, link, dataset)


def view_dataset():
    dataset = H5pyDatafile("C:/SpliceImageTextData/dataset/words.h5")
    count = {}

    def __view_dataset(i):
        nonlocal count
        decoded_str = i.decode("utf-8")
        str_count = len(decoded_str)
        if str_count <= 5:
            if str_count not in count:
                count[str_count] = [decoded_str]
            else:
                count[str_count].append(decoded_str)

    # 먼저 모든 데이터를 가져와서 count 딕셔너리를 채워줘야 해.
    dataset.get_all('word', __view_dataset)

    # 이제 count가 채워진 후 작업 시작
    current_counts = {k: len(v) for k, v in count.items()}
    max_count = max(current_counts.values())

    scaling_word = {}
    for length, words in count.items():
        current_count = len(words)
        if current_count < max_count:
            additional = random.choices(words, k=(max_count - current_count))
            scaling_word[length] = words + additional
        else:
            scaling_word[length] = words

    results = []
    for _, value in scaling_word.items():
        for item in value:
            results.append(item)

    for result in results:
        dataset.save("scaling_word", result)

    print(dataset.shape())


if __name__ == '__main__':
    view_dataset()