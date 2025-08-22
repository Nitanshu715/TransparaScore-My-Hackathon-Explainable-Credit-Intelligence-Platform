# 🏦 TransparaScore-My-Hackathon-Explainable-Credit-Intelligence-Platform

TransparaScore is a real-time explainable credit intelligence platform built for the **CredTech Hackathon**.  
It combines **financial data, macroeconomic indicators, and news events** to generate transparent creditworthiness scores for major companies.

---

## 🔗 [Live App](https://nitanshu-tak-transpara-score.streamlit.app/)


---

## 🚀 Features
- Live data from Yahoo Finance, FRED, World Bank, Reuters, and GDELT  
- Transparent scoring based on financial performance, macro conditions, and event sentiment  
- Factor contribution breakdowns and trend analysis  
- Interactive dashboard with charts, tables, and CSV export  
- Simple, rule-based system with clear explanations (no black box)

---

## 🛠 Tech Stack
- **Frontend & Dashboard**: Streamlit  
- **Data Processing**: Pandas, NumPy  
- **APIs & Sources**: Yahoo Finance, FRED, World Bank, GDELT, Reuters RSS  

---

## ▶️ Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TransparaScore.git
   cd TransparaScore
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Add your API keys in .streamlit/secrets.tom
   ```bash
   YAHOO_API_KEY = "YOUR_YAHOO_API_KEY"
   FRED_API_KEY = "YOUR_FRED_API_KEY"
   WORLDBANK_API = "http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json"
   GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
   RSS_FEED_URL = "https://feeds.reuters.com/reuters/businessNews"
4. Run the app:
   ```bash
   streamlit run TransparaScore.py

--- 

🌐 Deployment:
This app can be deployed on Streamlit Cloud by connecting this repo.
API keys should be stored securely in Streamlit Secrets.

---

⚠️ Disclaimer:
This project is built for hackathon and educational purposes only.
All data is fetched from public/free APIs and is not financial advice.
For demonstration of explainable credit intelligence.

---

## 🧑‍💻 Author

**Nitanshu Tak**  
- GitHub: [Nitanshu715](https://github.com/Nitanshu715)
- LinkedIn: [Nitanshu Tak](https://www.linkedin.com/in/nitanshu-tak-89a1ba289/)
- Email: nitanshutak070105@gmail.com

---

## 📜 License

MIT License — see `LICENSE` file for details.
