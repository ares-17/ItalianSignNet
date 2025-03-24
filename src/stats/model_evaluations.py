import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def error_distribution_by_area(data, output_dir):
    """Distribuzione delle previsioni corrette e sbagliate per area geografica"""
    
    results_by_geo = data.groupby(["Geographic Location", "Results"]).size().unstack(fill_value=0)
    results_by_geo["Total"] = results_by_geo.sum(axis=1)
    results_by_geo["% Correct (T)"] = (results_by_geo["T"] / results_by_geo["Total"]) * 100

    # Creazione di pie chart per ogni locazione geografica
    for location in results_by_geo.index:
        data_location = results_by_geo.loc[location, ["T", "F"]]
        labels = ["Correct (T)", "Incorrect (F)"]
        colors = ["#4CAF50", "#F44336"]

        plt.figure(figsize=(6, 6))
        plt.pie(
            data_location,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=(0.05, 0),
            wedgeprops={"edgecolor": "black"}
        )
        plt.title(f"Distribuzione previsioni - {location}")
        plt.tight_layout()
        filename = f"pie_chart_{location.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))

def average_probability_by_area(data, output_dir):
    """Probabilità media di previsione per area geografica e risultato"""
    
    probability_mean_geo = data.groupby(["Geographic Location", "Results"])["Probability"].mean().unstack()
    probability_mean_geo.plot(kind="bar", figsize=(10, 6), alpha=0.7)
    plt.title("Probabilità media per area geografica e risultato")
    plt.ylabel("Probabilità media")
    plt.xlabel("Area geografica")
    plt.legend(title="Risultati")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probabilita_media.png"))

def class_errors_by_area(data, output_dir):
    """Classi con più errori (F) per area geografica"""
    
    errors_by_class = data[data["Results"] == "F"].groupby(["Geographic Location", "Predicted Class"]).size().unstack(fill_value=0)
    errors_by_class.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    plt.title("Distribuzione delle classi con errori per area geografica")
    plt.ylabel("Numero di errori")
    plt.xlabel("Area geografica")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classi_con_errori.png"))

def best_class_by_answer(data, output_dir):
    """Classi più comuni riconosciute correttamente (T)"""
    
    correct_by_class = data[data["Results"] == "T"].groupby(["Geographic Location", "Predicted Class"]).size().unstack(fill_value=0)
    correct_by_class.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab10")
    plt.title("Distribuzione delle classi corrette per area geografica")
    plt.ylabel("Numero di riconoscimenti")
    plt.xlabel("Area geografica")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classi_correttamente_riconosciute.png"))

def performance_by_signals_features(data_performance, output_dir):
    """Performance basata sulle caratteristiche dei segnali"""

    for geo in data_performance.index.get_level_values(0).unique():
        geo_data = data_performance.loc[geo]
        geo_data[["% Correct (T)"]].sort_values(by="% Correct (T)", ascending=False).plot(kind="bar", figsize=(10, 6))
        plt.title(f"Performance basata sulle caratteristiche dei segnali - {geo}")
        plt.ylabel("Percentuale di correttezza")
        plt.xlabel("Caratteristiche dei segnali")
        plt.tight_layout()
        filename = f"performance_caratteristiche_{geo}.png"
        plt.savefig(os.path.join(output_dir, filename))

def total_performance_by_features(data_performance, output_dir):
    """Calcolo della performance complessiva (totale)"""
    
    total_performance = data_performance.groupby("feature").apply(
        lambda x: (x["T"].sum() / (x["T"].sum() + x["F"].sum())) * 100
    ).sort_values(ascending=False)

    total_performance.plot(kind="bar", figsize=(10, 6), color="#4CAF50")
    plt.title("Performance complessiva basata sulle caratteristiche dei segnali")
    plt.ylabel("Percentuale di correttezza")
    plt.xlabel("Caratteristiche dei segnali")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_caratteristiche_totale.png"))

def average_probability_by_class(data, output_dir):
    """Calcolo della probabilità media per classe"""
    
    prob_mean_by_class = data.groupby("Predicted Class")["Probability"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    prob_mean_by_class.plot(kind="bar", color="#4CAF50")
    plt.title("Probabilità media per classe")
    plt.ylabel("Probabilità media")
    plt.xlabel("Classe prevista")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probabilità_media_classe.png"))

def count_errors_by_class(data, output_dir):
    """Conteggio degli errori per classe"""
    
    errors_by_class = data[data["Results"] == "F"].groupby("Predicted Class").size().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    errors_by_class.plot(kind="bar", color="#F44336")
    plt.title("Numero di errori per classe")
    plt.ylabel("Numero di errori")
    plt.xlabel("Classe prevista")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "errori_per_classe.png"))

def outliers_by_probability(data, output_dir):
    """Box plot per identificare outlier nella probabilità"""
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="Results", y="Probability", hue="Results", palette={"T": "green", "F": "red"}, legend=False)
    plt.title("Analisi degli outlier - Probabilità per risultato")
    plt.ylabel("Probabilità")
    plt.xlabel("Risultato")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outlier_probabilità.png"))
    plt.close()

def results_distribution(data, output_dir):
    """Distribuzione delle probabilità per T e F"""
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x="Probability", hue="Results", kde=True, palette={"T": "green", "F": "red"}, alpha=0.5)
    plt.title("Distribuzione della probabilità per risultato")
    plt.xlabel("Probabilità")
    plt.ylabel("Densità")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribuzione_probabilità.png"))
    
def main():
    stats_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_dir = os.path.join(stats_directory, 'stats', '.results')
    os.makedirs(output_dir, exist_ok=True)

    csv_file = os.path.join(stats_directory, 'model', '.results','merged_file.csv')
    data = pd.read_csv(csv_file, sep=',')
    data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors="ignore")

    data_performance = data.groupby(["Geographic Location", "feature", "Results"]).size().unstack(fill_value=0)
    data_performance["% Correct (T)"] = (data_performance["T"] / (data_performance["T"] + data_performance["F"])) * 100
    
    error_distribution_by_area(data, output_dir)
    average_probability_by_area(data, output_dir)
    class_errors_by_area(data, output_dir)
    best_class_by_answer(data, output_dir)
    performance_by_signals_features(data_performance, output_dir)
    total_performance_by_features(data_performance, output_dir)
    average_probability_by_class(data, output_dir)
    count_errors_by_class(data, output_dir)
    outliers_by_probability(data, output_dir)
    results_distribution(data, output_dir)
    
if __name__ == '__main__':
    main()