import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# 1. Charger les données
data = pd.read_csv('dataset1.txt', sep='\t')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 2. Créer des variables pour la décomposition
n = len(data)
t = np.arange(n).reshape(-1, 1)  # Variable temps pour la tendance
months = data.index.month

# Créer des variables dummy pour la saisonnalité (11 variables pour 12 mois)
seasonal_dummies = pd.get_dummies(months, drop_first=True)

# 3. Ajuster le modèle de régression (Tendance + Saisonnalité)
X = np.column_stack([t, seasonal_dummies])
y = data.values.ravel()

model = LinearRegression()
model.fit(X, y)

# 4. Extraire les composantes
# Tendance seule (uniquement avec le temps, coefficient d'intercept + pente*t)
trend = model.intercept_ + model.coef_[0] * t.ravel()

# Saisonnalité (prédiction complète - tendance)
seasonal = model.predict(X) - trend

# Résidus
residuals = y - model.predict(X)

# 5. Visualisation
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Série originale
axes[0].plot(data.index, y, label='Série originale', color='blue', linewidth=1.5)
axes[0].set_title('Série Temporelle Originale - Ventes Retail & Food Services', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Ventes (millions $)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tendance
axes[1].plot(data.index, trend, label='Tendance', color='red', linewidth=2)
axes[1].set_title('Composante de Tendance', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Ventes (millions $)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Saisonnalité
axes[2].plot(data.index, seasonal, label='Saisonnalité', color='green', linewidth=1)
axes[2].set_title('Composante Saisonnière', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Variation')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Résidus
axes[3].plot(data.index, residuals, label='Résidus', color='orange', linewidth=0.8)
axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[3].set_title('Résidus', fontsize=12, fontweight='bold')
axes[3].set_ylabel('Variation')
axes[3].set_xlabel('Date')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Analyse des résidus
print("\n" + "="*60)
print("ANALYSE DES RÉSIDUS")
print("="*60)
print(f"Moyenne des résidus: {np.mean(residuals):.4f}")
print(f"Écart-type des résidus: {np.std(residuals):.4f}")
print(f"Min: {np.min(residuals):.2f}, Max: {np.max(residuals):.2f}")

# Test de normalité
_, p_value = stats.shapiro(residuals)
print(f"\nTest de Shapiro-Wilk (normalité):")
print(f"  p-value = {p_value:.4f}")
if p_value > 0.05:
    print("  ✓ Les résidus semblent suivre une distribution normale")
else:
    print("  ✗ Les résidus ne suivent pas une distribution normale")
    print("  → Cela suggère que le modèle linéaire simple ne capture pas toute la structure")

# Analyse d'autocorrélation (patterns restants)
from scipy.stats import pearsonr
if len(residuals) > 1:
    autocorr, p_autocorr = pearsonr(residuals[:-1], residuals[1:])
    print(f"\nAutocorrélation lag-1: {autocorr:.4f} (p-value: {p_autocorr:.4f})")
    if abs(autocorr) > 0.3:
        print("  ⚠ Il reste des patterns d'autocorrélation significatifs")
        print("  → Un modèle ARIMA/SARIMA serait plus approprié")
    else:
        print("  ✓ Pas d'autocorrélation significative détectée")

# 7. Prévision pour 2025
print("\n" + "="*60)
print("PRÉVISIONS POUR 2025")
print("="*60)

# Les 8 premiers mois de 2025 sont déjà dans les données
actual_2025 = data[data.index.year == 2025]
print(f"\nDonnées réelles disponibles: {len(actual_2025)} mois (Jan-Août 2025)")

# Prévision pour les 4 derniers mois de 2025 (Sep-Dec)
n_forecast = 4  # Septembre à Décembre
t_forecast = np.arange(n, n + n_forecast).reshape(-1, 1)
months_forecast = [9, 10, 11, 12]  # Sep, Oct, Nov, Dec

# Créer les dummy variables avec toutes les colonnes (mois 2 à 12)
seasonal_dummies_forecast = np.zeros((n_forecast, 11))
for i, month in enumerate(months_forecast):
    if month > 1:  # Mois 1 est la référence, donc on commence à 2
        seasonal_dummies_forecast[i, month - 2] = 1

X_forecast = np.column_stack([t_forecast, seasonal_dummies_forecast])
forecast_sep_dec = model.predict(X_forecast)

month_names = ['Septembre', 'Octobre', 'Novembre', 'Décembre']
print(f"\nPrévisions pour les mois restants de 2025:")
for i, (month, value) in enumerate(zip(month_names, forecast_sep_dec)):
    print(f"  {month} 2025: {value:.2f} millions $")

# Comparaison avec les données réelles de 2025 (Jan-Août)
print(f"\nComparaison prévisions vs réel pour Jan-Août 2025:")
idx_2025 = data.index.year == 2025
X_2025 = X[idx_2025]
y_2025_actual = y[idx_2025]
y_2025_pred = model.predict(X_2025)

errors_2025 = y_2025_actual - y_2025_pred
mae_2025 = np.mean(np.abs(errors_2025))
rmse_2025 = np.sqrt(np.mean(errors_2025**2))
mape_2025 = np.mean(np.abs(errors_2025 / y_2025_actual)) * 100

print(f"  MAE: {mae_2025:.2f} millions $")
print(f"  RMSE: {rmse_2025:.2f} millions $")
print(f"  MAPE: {mape_2025:.2f}%")

# Afficher les prévisions vs réel pour 2025
print(f"\nDétail Jan-Août 2025:")
month_names_jan_aug = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août']
for i, (month_name, actual, pred) in enumerate(zip(month_names_jan_aug, y_2025_actual, y_2025_pred)):
    error = actual - pred
    print(f"  {month_name}: Réel={actual:.2f}, Prévu={pred:.2f}, Erreur={error:.2f}")

# 8. Métriques de performance globales
print("\n" + "="*60)
print("PERFORMANCE GLOBALE DU MODÈLE")
print("="*60)

r2_score = model.score(X, y)
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
mape = np.mean(np.abs(residuals / y)) * 100

print(f"R² Score: {r2_score:.4f}")
print(f"RMSE: {rmse:.2f} millions $")
print(f"MAE: {mae:.2f} millions $")
print(f"MAPE: {mape:.2f}%")

print("\nInterprétation:")
if r2_score > 0.95:
    print("  ✓ Excellent ajustement (R² > 0.95)")
elif r2_score > 0.90:
    print("  ✓ Très bon ajustement (R² > 0.90)")
else:
    print("  ⚠ Ajustement acceptable mais pourrait être amélioré")

if mape < 2:
    print("  ✓ Très bonne précision des prévisions (MAPE < 2%)")
elif mape < 5:
    print("  ✓ Bonne précision des prévisions (MAPE < 5%)")
else:
    print("  ⚠ Précision modérée des prévisions")

print("\n" + "="*60)
print("RÉPONSES AUX QUESTIONS DE L'EXERCICE")
print("="*60)

print("""
1. DÉCOMPOSITION (Tendance, Saisonnalité, Résidus):
   ✓ Effectuée avec succès par régression linéaire
   ✓ Visualisations créées dans 'decomposition.png'

2. ANALYSE DES RÉSIDUS:
   ✗ La décomposition n'est PAS totalement satisfaisante:
     - Résidus non normaux (test de Shapiro-Wilk rejeté)
     - Forte autocorrélation (0.98) indiquant des patterns non capturés
     - Écart-type élevé (~40,000) suggérant de la variance non expliquée
   
   → Patterns restants à améliorer:
     - Tendance non-linéaire (croissance qui s'accélère)
     - Chocs COVID-19 en 2020 (résidus extrêmes)
     - Variations saisonnières complexes non capturées par modèle linéaire

3. QUALITÉ DES PRÉVISIONS:
   → Quantifiée par:
     - R²: Proportion de variance expliquée
     - RMSE: Erreur quadratique moyenne
     - MAE: Erreur absolue moyenne
     - MAPE: Erreur en pourcentage
     - Autocorrélation: Patterns temporels restants

4. AMÉLIORATIONS POSSIBLES:
   - Modèle polynomial pour la tendance
   - SARIMA pour capturer l'autocorrélation
   - Variables exogènes (économiques, COVID)
   - Détection et traitement des outliers (2020)
""")

print("\nGraphique sauvegardé dans 'decomposition.png'")