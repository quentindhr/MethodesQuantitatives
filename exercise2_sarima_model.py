import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXERCICE 2 - MODÉLISATION SARIMA")
print("Dataset: Electric Power Consumption") 
print("="*70)


print("\n1. EXPLORATION DES DONNÉES")
print("-"*70)



data = pd.read_csv('dataset2.txt', sep='\t', parse_dates=['Date'])
data.set_index('Date', inplace=True)

print(f"Période: {data.index.min()} à {data.index.max()}")
print(f"Nombre d'observations: {len(data)}")
print(f"\nPremières lignes:")
print(data.head())
print(f"\nStatistiques descriptives:")
print(data.describe())


missing = data.isnull().sum()
print(f"\nValeurs manquantes: {missing.values[0]}")
if missing.values[0] > 0:
    print("  → Interpolation des valeurs manquantes...")
    data = data.interpolate(method='linear')


print("\n" + "="*70)
print("2. ANALYSE DE STATIONNARITÉ")
print("-"*70)

consumption = data.iloc[:, 0]

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    print(f"\nTest ADF pour {name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Valeurs critiques:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.3f}")
    
    if result[1] <= 0.05:
        print("  ✓ Série STATIONNAIRE (p-value ≤ 0.05)")
        return True
    else:
        print("  ✗ Série NON-STATIONNAIRE (p-value > 0.05)")
        return False

is_stationary = adf_test(consumption, "Série originale")

# Différenciation si nécessaire
if not is_stationary:
    print("\n→ Application de la différenciation...")
    diff1 = consumption.diff().dropna()
    is_stationary_diff1 = adf_test(diff1, "Différence d'ordre 1")
    
    if not is_stationary_diff1:
        print("\n→ Application d'une seconde différenciation...")
        diff2 = diff1.diff().dropna()
        adf_test(diff2, "Différence d'ordre 2")
        d_param = 2
    else:
        d_param = 1
else:
    d_param = 0

print(f"\n→ Paramètre de différenciation recommandé: d = {d_param}")


print("\n→ Test de stationnarité saisonnière...")
seasonal_diff = consumption.diff(12).dropna()
is_seasonal_stationary = adf_test(seasonal_diff, "Différence saisonnière (lag=12)")

if not is_seasonal_stationary:
    D_param = 1
else:
    D_param = 0

print(f"→ Paramètre de différenciation saisonnière recommandé: D = {D_param}")


print("\n" + "="*70)
print("3. IDENTIFICATION DES PARAMÈTRES SARIMA")
print("-"*70)


if d_param > 0:
    series_for_acf = consumption.diff(d_param).dropna()
else:
    series_for_acf = consumption


fig, axes = plt.subplots(2, 2, figsize=(14, 8))


axes[0, 0].plot(consumption)
axes[0, 0].set_title('Série Originale - Consommation Électrique')
axes[0, 0].set_ylabel('Consommation')
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].plot(series_for_acf)
axes[0, 1].set_title(f'Série Différenciée (d={d_param})')
axes[0, 1].set_ylabel('Différence')
axes[0, 1].grid(True, alpha=0.3)


plot_acf(series_for_acf, lags=40, ax=axes[1, 0])
axes[1, 0].set_title('Autocorrelation Function (ACF)')


plot_pacf(series_for_acf, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.savefig('sarima_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n→ Graphiques ACF/PACF sauvegardés dans 'sarima_acf_pacf.png'")

print("\nInterprétation des graphiques pour identifier (p, d, q)(P, D, Q)s:")
print("  - ACF: Pics significatifs → paramètre q (MA)")
print("  - PACF: Pics significatifs → paramètre p (AR)")
print("  - Pics saisonniers (lag 12, 24, ...) → P, Q")


print("\n→ Paramètres suggérés à tester:")
print(f"  Non-saisonnier: p=1, d={d_param}, q=1")
print(f"  Saisonnier: P=1, D={D_param}, Q=1, s=12")


print("\n" + "="*70)
print("4. CONSTRUCTION DU MODÈLE SARIMA")
print("-"*70)


train_size = int(len(consumption) * 0.8)
train = consumption[:train_size]
test = consumption[train_size:]

print(f"Données d'entraînement: {len(train)} observations")
print(f"Données de test: {len(test)} observations")


configs = [
    ((1, d_param, 1), (1, D_param, 1, 12)),
    ((1, d_param, 0), (1, D_param, 1, 12)),
    ((0, d_param, 1), (1, D_param, 1, 12)),
    ((2, d_param, 2), (1, D_param, 1, 12)),
]

print("\n→ Test de plusieurs configurations SARIMA...")
best_aic = np.inf
best_config = None
best_model = None

for order, seasonal_order in configs:
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
        aic = fitted_model.aic
        print(f"  SARIMA{order}x{seasonal_order}: AIC={aic:.2f}")
        
        if aic < best_aic:
            best_aic = aic
            best_config = (order, seasonal_order)
            best_model = fitted_model
    except Exception as e:
        print(f"  SARIMA{order}x{seasonal_order}: Erreur - {str(e)[:50]}")

print(f"\n✓ Meilleur modèle: SARIMA{best_config[0]}x{best_config[1]}")
print(f"  AIC: {best_aic:.2f}")

print("\n→ Résumé du modèle:")
print(best_model.summary())


print("\n" + "="*70)
print("5. ÉVALUATION DU MODÈLE")
print("-"*70)


predictions = best_model.forecast(steps=len(test))


mae = np.mean(np.abs(test.values - predictions))
rmse = np.sqrt(np.mean((test.values - predictions)**2))
mape = np.mean(np.abs((test.values - predictions) / test.values)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")


residuals = best_model.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 8))


axes[0, 0].plot(residuals)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Résidus du Modèle')
axes[0, 0].set_ylabel('Résidus')
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_title('Distribution des Résidus')
axes[0, 1].set_xlabel('Résidus')
axes[0, 1].set_ylabel('Fréquence')


plot_acf(residuals, lags=30, ax=axes[1, 0])
axes[1, 0].set_title('ACF des Résidus')


from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('sarima_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n→ Analyse des résidus sauvegardée dans 'sarima_residuals.png'")


print("\n" + "="*70)
print("6. PRÉVISIONS ET VISUALISATION")
print("-"*70)

n_forecast = 12
future_forecast = best_model.forecast(steps=len(test) + n_forecast)


fig, ax = plt.subplots(figsize=(14, 6))


ax.plot(train.index, train.values, label='Entraînement', color='blue', linewidth=1.5)


ax.plot(test.index, test.values, label='Test (Réel)', color='green', linewidth=1.5)


ax.plot(test.index, predictions, label='Prédictions Test', 
        color='orange', linewidth=1.5, linestyle='--')


future_dates = pd.date_range(start=test.index[-1], periods=n_forecast+1, freq='MS')[1:]
ax.plot(future_dates, future_forecast[len(test):], label='Prévisions Futures', 
        color='red', linewidth=2, linestyle='--')

ax.set_title(f'Modèle SARIMA{best_config[0]}x{best_config[1]} - Prévisions', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Consommation Électrique')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sarima_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n→ Prévisions sauvegardées dans 'sarima_forecast.png'")

print("\n→ Prévisions pour les 12 prochains mois:")
for i, (date, value) in enumerate(zip(future_dates, future_forecast[len(test):])):
    print(f"  {date.strftime('%Y-%m')}: {value:.2f}")


print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
✓ Modèle SARIMA{best_config[0]}x{best_config[1]} ajusté avec succès

PERFORMANCES:
- MAE: {mae:.2f}
- RMSE: {rmse:.2f}
- MAPE: {mape:.2f}%

POINTS CLÉS:
1. Stationnarité obtenue avec d={d_param} différenciation(s)
2. Composante saisonnière capturée avec période s=12
3. Prévisions générées pour {n_forecast} mois futurs
4. Résidus analysés pour valider le modèle

AMÉLIORATIONS POSSIBLES:
- Tester d'autres ordres (p, q, P, Q)
- Ajouter des variables exogènes
- Utiliser des modèles plus complexes (Prophet, LSTM)
""")

print("\nFichiers générés:")
print("  - sarima_acf_pacf.png")
print("  - sarima_residuals.png")
print("  - sarima_forecast.png")