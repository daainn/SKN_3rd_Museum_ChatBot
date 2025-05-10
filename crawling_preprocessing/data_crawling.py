import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import time


service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument('--headless')  
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)  

url = 'https://www.museum.go.kr/site/main/relic/search/collectionList#######21480'

total_pages = 500 

csv_filename = "museum_data_1600.csv"

data = []

def extract_data(item, page, i):
    try:
        title_element = item.find_element(By.CSS_SELECTOR, "div > a")
        title = title_element.text.strip()

        img_element = item.find_element(By.CSS_SELECTOR, "a > img")
        img_url = img_element.get_attribute("src")

        driver.execute_script("arguments[0].click();", title_element)

        description_element = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#contents-area > div.page-content-type2 > div.page-content-type1 > div > div > div.outview-txt-box.collection > div > div.view-info-cont.view-info-cont2 > p"))
        )
        description = description_element.text.strip()

        return [title, img_url, description]

    except (StaleElementReferenceException, TimeoutException) as e:
        print(f"오류 발생 (페이지 {page}, 항목 {i}): {e}")
        return None
    except Exception as e:
        print(f"예기치 않은 오류 발생 (페이지 {page}, 항목 {i}): {e}")
        return None

try:
    driver.get(url)
    current_page = 1

    while current_page <= total_pages:
        print(f"페이지 {current_page}")

        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        try:
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#card1 > li")))
        except TimeoutException:
            print(f"페이지 {current_page} 로딩 시간 초과")
            continue

        items = driver.find_elements(By.CSS_SELECTOR, "#card1 > li")
        for i, item in enumerate(items, 1):
            extracted = extract_data(item, current_page, i)
            if extracted:
                data.append(extracted)
            
            driver.back()
            wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#card1 > li")))

        if current_page % 10 == 0 and current_page < total_pages:
            try:
                next_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#contents-area > div.page-content-type1 > div > div.board-list-foot > div > div.pagenation > a.next > img"))
                )
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(5) 
            except TimeoutException:
                print(f"페이지 {current_page}에서 다음 버튼을 찾을 수 없습니다.")
                break
        else:
            try:
                next_page_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, f"#contents-area > div.page-content-type1 > div > div.board-list-foot > div > div.pagenation > ul > li:nth-child({(current_page % 10) + 1}) > a"))
                )
                driver.execute_script("arguments[0].click();", next_page_button)
                time.sleep(5)  
            except TimeoutException:
                print(f"페이지 {current_page}에서 다음 페이지 버튼을 찾을 수 없습니다.")
                break

        current_page += 1

except Exception as e:
    print(f"예기치 않은 최상위 오류 발생: {e}")

finally:
    driver.quit() 


try:
    with open(csv_filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Image URL", "Description"])
        writer.writerows(data)
    print(f"파일명: {csv_filename}")
    print(f"{len(data)}개의 항목이 저장되었습니다.")

except Exception as e:
    print(f"CSV 파일 저장 중 오류 발생: {e}")
