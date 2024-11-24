from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

# ChromeDriver-Pfad
driver_path = "C:/Program Files/Google/Chrome/Application/chromedriver.exe"  # Passen Sie diesen Pfad an Ihren Systempfad an

# ChromeDriver-Service erstellen
service = Service(driver_path)
driver = webdriver.Chrome(service=service)

# URL der Seite
url = "https://finance.yahoo.com/quote/AMZN/history/?period1=1574369042&period2=1732221002"
driver.get(url)

# Zeit zum Laden der Inhalte geben
time.sleep(5)

# Tabelle finden
try:
    # Table-Element auswählen
    table = driver.find_element(By.XPATH, '//div[@data-testid="history-table"]//table')

    # Zeilen der Tabelle extrahieren
    rows = table.find_elements(By.TAG_NAME, "tr")
    data = []

    # Jede Zeile und Spalte lesen
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        cols = [ele.text for ele in cols]
        if cols:  # Nur nicht-leere Zeilen speichern
            data.append(cols)

    # DataFrame erstellen
    df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close*", "Adj Close**", "Volume"])

    # CSV speichern
    df.to_csv("amazon_historical_prices_dynamic.csv", index=False)
    print("Daten wurden in 'amazon_historical_prices_dynamic.csv' gespeichert")

except Exception as e:
    print("Fehler beim Abrufen der Tabelle:", e)

# Browser schließen
driver.quit()

