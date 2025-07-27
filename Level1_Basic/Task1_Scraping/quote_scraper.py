import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL
base_url = "http://quotes.toscrape.com/page/{}/"

# Store scraped data
quotes = []
authors = []
tags_list = []

# Loop through multiple pages
for page in range(1, 11):  # The site has 10 pages
    print(f"Scraping page {page}...")
    response = requests.get(base_url.format(page))
    soup = BeautifulSoup(response.content, "html.parser")

    quote_blocks = soup.find_all("div", class_="quote")
    
    for quote_block in quote_blocks:
        quote = quote_block.find("span", class_="text").text
        author = quote_block.find("small", class_="author").text
        tags = [tag.text for tag in quote_block.find_all("a", class_="tag")]

        quotes.append(quote)
        authors.append(author)
        tags_list.append(", ".join(tags))

# Create DataFrame
df = pd.DataFrame({
    "Quote": quotes,
    "Author": authors,
    "Tags": tags_list
})

# Save to CSV
df.to_csv("data/quotes.csv", index=False)
print("✅ Scraping complete. Data saved to data/quotes.csv")
