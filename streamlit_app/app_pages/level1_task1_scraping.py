import sys
import os
import glob
import pandas as pd
import streamlit as st

from Level1_Basic.Task1_Scraping.scraper import scrape_quotes, search_quotes, is_dynamic, extract_structured_content, save_scraped_text_to_csv

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def main():    
    st.set_page_config(page_title="Web Scraping | Level 1 - Task 1", layout="wide")

    st.title("🕸️ Web Scraping - Level 1 Task 1")

    tab1, tab2 = st.tabs(["🌐 Built-in Quote Scraper", "🔗 Custom URL Scraper"])

    # --- TAB 1: Built-in Quotes Scraper ---
    with tab1:
        st.subheader("🌐 Scrape from Demo Site (http://quotes.toscrape.com)")
        st.markdown("This is a test-friendly site built for scraping practice.")

        if st.button("🚀 Scrape Demo Quotes"):
            with st.spinner("Scraping quotes..."):
                df = scrape_quotes()
                st.success(f"✅ Scraped {len(df)} quotes.")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", csv, "quotes.csv", "text/csv")

        st.markdown("## 🔍 Search Quotes")
        search_term = st.text_input("Enter keyword (quote, author, or tag):", key="search1")
        if search_term:
            if 'df' not in locals():
                df = scrape_quotes()
            results = search_quotes(df, search_term)
            st.markdown(f"### Found {len(results)} result(s):")
            st.dataframe(results, use_container_width=True)
    
    # --- TAB 2: Custom URL Scraper & Loader ---
    with tab2:
        st.subheader("🔗 Custom URL Scraper & Page Loader")

        st.markdown("### 📂 Load Previously Scraped File")
        file_list = glob.glob("data/raw/*_scraped.csv")
        selected_file = st.selectbox("Select a saved file to preview:", [""] + file_list)

        if selected_file and selected_file != "":
            try:
                df_loaded = pd.read_csv(selected_file)
                st.success(f"✅ Loaded {selected_file}")
                # Toggle between views
                view_option = st.radio(
                    "Choose preview mode:",
                    ("Markdown Preview", "Raw CSV Data")
                )

                if view_option == "Markdown Preview":
                    st.markdown("### 📄 Markdown Preview of Saved Content")
                    text = df_loaded["text"].iloc[0] if "text" in df_loaded.columns else None
                    if text:
                        st.markdown(text)
                    else:
                        st.warning("⚠️ No 'text' column found in the selected file.")
                else:
                    st.markdown("### 🧾 Raw CSV Data")
                    st.dataframe(df_loaded, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")

        st.markdown("---")
        st.markdown("### 🌍 Scrape New Custom URL")
        custom_url = st.text_input("Enter a new URL to scrape:")

        if custom_url:
            if st.button("🔍 Extract Visible Text Content"):
                with st.spinner("Extracting visible text..."):
                    try:
                        text_content = extract_structured_content(custom_url)

                        if text_content:
                            st.success("✅ Text extracted successfully!")

                            st.markdown("### 📄 Page Preview (First 1000 characters)")
                            st.markdown(f"```markdown\n{text_content[:1000]}\n```")

                            with st.expander("📖 Show Full Extracted Content"):
                                st.markdown(text_content)

                            if st.button("📁 Save as CSV for Preprocessing"):
                                try:
                                    output_path = save_scraped_text_to_csv(custom_url, text_content)
                                    st.success(f"✅ Saved to `{output_path}`")
                                except Exception as e:
                                    st.error(str(e))

                            st.download_button(
                                label="💾 Download Full Text as .txt",
                                data=text_content,
                                file_name="scraped_content.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("⚠️ No visible text found on the page.")
                    except Exception as e:
                        st.error(f"❌ Failed to extract content: {e}")