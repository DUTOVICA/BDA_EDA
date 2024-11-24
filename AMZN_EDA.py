import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# CSV-Datei laden
file_path = "amazon_historical_prices_dynamic.csv"
df = pd.read_csv(file_path)

# Spalten filtern
df = df[["Date", "Open", "Close*"]]

# Spaltennamen bereinigen
df.columns = ["Date", "Open", "Close"]

# Sicherstellen, dass 'Open' und 'Close' als Strings behandelt werden
df["Open"] = df["Open"].astype(str).str.replace(",", "", regex=False)
df["Close"] = df["Close"].astype(str).str.replace(",", "", regex=False)

# Datentypen konvertieren
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Fehlende Werte entfernen
df = df.dropna()

# Überblick über die Daten
print("Erste Zeilen des Datensatzes:")
print(df.head())

print("\nStatistische Beschreibung:")
print(df.describe())

# Interaktive Zeitreihenanalyse mit Plotly
fig = go.Figure()

# Open und Close auf der Zeitachse
fig.add_trace(go.Scatter(x=df["Date"], y=df["Open"], mode='lines', name="Open", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode='lines', name="Close", line=dict(color='orange')))

fig.update_layout(
    title="Amazon Open vs Close Prices Over Time",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    hovermode="closest"
)

fig.show()

# Verteilung der 'Open'- und 'Close'-Preise mit Plotly
fig = px.histogram(df, x="Open", nbins=50, title="Distribution of Open Prices", opacity=0.6, color_discrete_sequence=["blue"])
fig.add_histogram(x=df["Close"], nbinsx=50, opacity=0.6, name="Close Prices", marker=dict(color="orange"))
fig.update_layout(
    xaxis_title="Price",
    yaxis_title="Frequency",
    barmode='overlay',
    template="plotly_dark"
)

fig.show()


# Korrelation zwischen Open und Close
correlation = df["Open"].corr(df["Close"])
print(f"\nKorrelation zwischen Open und Close: {correlation:.2f}")

# Interaktive Heatmap der Korrelation mit Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=1)
plt.title("Correlation Heatmap")
plt.show()

# Moderne Pairplot mit Seaborn für Open und Close
sns.pairplot(df[["Open", "Close"]], kind="scatter", height=3, plot_kws={'alpha':0.7})
plt.suptitle("Pairplot of Open and Close Prices", y=1.02)
plt.show()
