# 🛍️ Product Price Finder BOT

This is a lightweight LLM-powered assistant that extracts the **Product Name** and its **Tentative Price** from a product description. It uses **Groq LLMs** (via [Langchain](https://www.langchain.com/)) and provides an easy-to-use **Streamlit UI**. Also, tested few **Google LLM's** model's as well.

---

## 🚀 Features

-  Uses multiple **Groq-hosted large language models**
-  Extracts structured product name and estimated price (USD)
-  Built with **Pydantic** to validate output format
-  Interactive **Streamlit UI**
-  Prompt designed for reliability and clarity

---

##  Requirements

Make sure you have Python 3.10+ installed.

Install dependencies:

```bash
pip install streamlit langchain_groq pydantic python-dotenv langchain_google_genai

