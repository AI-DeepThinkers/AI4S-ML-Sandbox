
import os
import re
import time
import requests
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def is_dynamic(url="http://quotes.toscrape.com"):
    headers = {"User-Agent": "Mozilla/5.0"}
    static_resp = requests.get(url, headers=headers)
    static_len = len(static_resp.text)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    selenium_len = len(driver.page_source)
    driver.quit()

    return selenium_len - static_len > 500

def scrape_quotes(base_url="http://quotes.toscrape.com"):
    all_quotes = []
    page = 1

    while True:
        url = f"{base_url}/page/{page}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        quotes_divs = soup.find_all("div", class_="quote")
        if not quotes_divs:
            break

        for quote_div in quotes_divs:
            text = quote_div.find("span", class_="text").get_text(strip=True)
            author = quote_div.find("small", class_="author").get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in quote_div.find_all("a", class_="tag")]
            all_quotes.append({"Quote": text, "Author": author, "Tags": ", ".join(tags)})

        page += 1
        time.sleep(0.5)

    return pd.DataFrame(all_quotes)

def search_quotes(df, keyword):
    keyword = keyword.lower()
    filtered = df[df["Quote"].str.lower().str.contains(keyword) | 
                  df["Author"].str.lower().str.contains(keyword) | 
                  df["Tags"].str.lower().str.contains(keyword)]
    return filtered

def extract_structured_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Elements we want to extract
    content_elements = soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "li"])

    markdown_output = ""
    for tag in content_elements:
        text = tag.get_text(strip=True)

        if not text:
            continue

        if tag.name == "h1":
            markdown_output += f"# {text}\n\n"
        elif tag.name == "h2":
            markdown_output += f"## {text}\n\n"
        elif tag.name == "h3":
            markdown_output += f"### {text}\n\n"
        elif tag.name == "p":
            markdown_output += f"{text}\n\n"
        elif tag.name in ["ul", "ol"]:
            continue  # Handled by <li>
        elif tag.name == "li":
            markdown_output += f"- {text}\n"

    return markdown_output.strip()

def save_scraped_text_to_csv(url, text_content, save_dir="data/raw"):
    """
    Save the scraped text content to a CSV file with a filename derived from the page title.

    Parameters:
        url (str): URL of the scraped page
        text_content (str): Cleaned text content
        save_dir (str): Directory to save CSV (default: data/raw)

    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Fetch HTML to get <title>
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.title.string if soup.title else "custom"
        clean_title = re.sub(r"[^\w\s-]", "", title_tag).strip().lower().replace(" ", "_")
        filename = f"{clean_title}_scraped.csv"

        # Save text content as DataFrame
        df = pd.DataFrame({"text": [text_content]})
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)
        df.to_csv(output_path, index=False)

        return output_path
    except Exception as e:
        raise RuntimeError(f"Error saving scraped CSV: {e}")

