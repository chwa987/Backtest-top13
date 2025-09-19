import datetime
import streamlit as st

# 📅 Datumsauswahl mit Defaultwerten
start_date = st.date_input(
    "Startdatum", 
    datetime.date(2018, 1, 1)   # Default: 1.1.2018
)

end_date = st.date_input(
    "Enddatum", 
    datetime.date.today()       # Default: Heute
)

# 🔄 Umwandeln in datetime.datetime für yfinance & Backtest
start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
