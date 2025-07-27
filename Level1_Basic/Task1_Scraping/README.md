# Level 1 Task 1: Web Scraping

Welcome to **Task 1** a project focused on collecting and extracting structured data from websites using web scraping techniques.

---

## 📖 Overview

This task demonstrates how to:

- Detect if a page is **static or dynamic**
- Scrape data from a default site (`http://quotes.toscrape.com`)
- Allow users to input a custom URL and extract readable content
- Display extracted data using **Streamlit** with a clean UI

---

## 🚀 Features

- ✅ **Built-in Demo Scraper**: Collects quotes, authors, and tags from a static site
- ✅ **Custom URL Support**: Accepts any user-defined URL and:
  - Checks if the page is static or dynamic
  - Extracts visible readable content (titles, paragraphs, lists)
- ✅ **Heuristic Page Type Detection** using length differences between raw HTML and Selenium-rendered DOM
- ✅ **Markdown-formatted output** for improved readability
- ✅ **Polite Scraping** using delay and user-agent headers

---

## 🧰 Tech Stack

- `Python`
- `BeautifulSoup`
- `requests`
- `Selenium`
- `pandas`
- `Streamlit`

---

## 🗂️ Project Structure

```bash
root/
│
├── Level1_Basic/
│   └── Task1_Scraping/
│       ├── scraper.py              # Core scraping logic
│       └── sample_outputs/         # Example scraped CSV/JSON files
│
├── streamlit_app/
│   ├── app_pages/
│   │   └── level1_task1_scraping.py   # Streamlit UI
│   └── streamlit_app.py               # Main multi-page launcher (optional)
│
├── data/
│   └── quotes.csv / quotes.json       # Output files
│
├── README.md
└── requirements.txt

-- 
## 📦 Setup Instructions

1. Clone the repository


2. Create and activate a virtual environment
bash
python -m venv
env\Scripts\activate     # Windows
source env/bin/activate # macOS/Linux

3. Install dependencies
bash
pip install -r requirements.txt

4. Launch Streamlit app
bash
cd streamlit_app/app_pages
streamlit run level1_task1_scraping.py


## 🧪 Demo Site Used
- URL: http://quotes.toscrape.com
- Public, static, and intentionally built for scraping practice

## 📤 Outputs
- quotes.csv and quotes.json saved in /data/
- Cleaned and tagged content from user-defined URLs shown in browser

## 🧠 Learnings
This task taught foundational web scraping skills and introduced ethical scraping practices and how to work with both static and dynamic websites.

📸 Preview
<!-- Optional: Include a screenshot -->

## 🔖 Hashtags for LinkedIn
txt
| #WebScraping #Python #Streamlit #BeautifulSoup #InternshipProject


