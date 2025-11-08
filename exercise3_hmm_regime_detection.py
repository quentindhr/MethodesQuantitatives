import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EXERCICE 3 - HIDDEN MARKOV MODEL (HMM)")
print("Dataset: S&P 500 Index - Détection de Régimes de Marché")
print("="*70)


print("\n1. CHARGEMENT ET PRÉPARATION DES DONNÉES")
print("-"*70)


data = pd.read_excel('dataset3.xlsx', parse_dates=['Date']) 
data.set_index('Date', inplace=True)
print("✓ Données chargées depuis dataset3.xlsx")


print(f"\nPériode: {data.index.min().date()} à {data.index.max().date()}")
print(f"Nombre d'observations: {len(data)}")
print(f"\nPremières lignes:")
print(data.head())


print("\n2. CALCUL DES RENDEMENTS")
print("-"*70)


data['Returns'] = data['Close'].pct_change()
data = data.dropna()

print(f"Statistiques des rendements:")
print(data['Returns'].describe())

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(data.index, data['Close'], color='blue', linewidth=1)
axes[0].set_title('S&P 500 - Prix de Clôture', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Prix')
axes[0].grid(True, alpha=0.3)

axes[1].plot(data.index, data['Returns'], color='gray', linewidth=0.5, alpha=0.7)
axes[1].set_title('Rendements Journaliers', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Rendement')
axes[1].set_xlabel('Date')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hmm_data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGraphique sauvegardé dans 'hmm_data_exploration.png'")


print("\n" + "="*70)
print("3. AJUSTEMENT DU MODÈLE HMM")
print("-"*70)


returns_array = data['Returns'].values.reshape(-1, 1)


n_states_list = [2, 3]
models = {}
scores = {}

for n_states in n_states_list:
    print(f"\nTest avec {n_states} états cachés...")
    

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    
    model.fit(returns_array)
    

    score = model.score(returns_array)
    scores[n_states] = score
    models[n_states] = model
    
    print(f"  Log-likelihood: {score:.2f}")
    print(f"  AIC: {-2 * score + 2 * (n_states**2 + 2*n_states):.2f}")
    print(f"  BIC: {-2 * score + (n_states**2 + 2*n_states) * np.log(len(returns_array)):.2f}")


print(f"\nModèle sélectionné: {2} états (interprétation financière plus claire)")
best_model = models[2]
n_states = 2


print("\n" + "="*70)
print("4. IDENTIFICATION DES RÉGIMES")
print("-"*70)


hidden_states = best_model.predict(returns_array)


data['Regime'] = hidden_states


print("\nCaractéristiques des régimes détectés:")
print("-"*70)

regime_stats = []
for state in range(n_states):
    regime_data = data[data['Regime'] == state]['Returns']
    
    mean_return = regime_data.mean()
    std_return = regime_data.std()
    count = len(regime_data)
    pct = (count / len(data)) * 100
    
    regime_stats.append({
        'State': state,
        'Mean': mean_return,
        'Std': std_return,
        'Count': count,
        'Percentage': pct
    })
    
    print(f"\nRÉGIME {state}:")
    print(f"  Rendement moyen: {mean_return*100:.4f}% par jour")
    print(f"  Rendement annualisé: {mean_return*252*100:.2f}%")
    print(f"  Volatilité: {std_return*100:.4f}%")
    print(f"  Volatilité annualisée: {std_return*np.sqrt(252)*100:.2f}%")
    print(f"  Nombre de jours: {count} ({pct:.1f}%)")


regime_stats_df = pd.DataFrame(regime_stats)
regime_stats_df = regime_stats_df.sort_values('Mean', ascending=False)

bull_state = regime_stats_df.iloc[0]['State']
bear_state = regime_stats_df.iloc[1]['State']

print(f"\nINTERPRÉTATION:")
print(f"  RÉGIME {int(bull_state)}: MARCHÉ HAUSSIER (Bull Market)")
print(f"  RÉGIME {int(bear_state)}: MARCHÉ BAISSIER (Bear Market)")


data['Regime_Label'] = data['Regime'].map({
    bull_state: 'Bull',
    bear_state: 'Bear'
})


print("\n" + "="*70)
print("5. VISUALISATION DES RÉGIMES")
print("-"*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))


for state, label, color in [(bull_state, 'Bull Market', 'green'), 
                             (bear_state, 'Bear Market', 'red')]:
    mask = data['Regime'] == state
    axes[0].scatter(data.index[mask], data['Close'][mask], 
                   c=color, s=1, alpha=0.6, label=label)

axes[0].set_title('S&P 500 - Prix avec Régimes Détectés', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Prix')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rendements avec régimes
for state, label, color in [(bull_state, 'Bull Market', 'green'), 
                             (bear_state, 'Bear Market', 'red')]:
    mask = data['Regime'] == state
    axes[1].scatter(data.index[mask], data['Returns'][mask], 
                   c=color, s=1, alpha=0.5, label=label)

axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[1].set_title('Rendements avec Régimes', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Rendement')
axes[1].legend()
axes[1].grid(True, alpha=0.3)


regime_numeric = data['Regime'].copy()
regime_numeric[regime_numeric == bull_state] = 1
regime_numeric[regime_numeric == bear_state] = -1

axes[2].fill_between(data.index, 0, regime_numeric, 
                     where=(regime_numeric > 0), color='green', alpha=0.3, label='Bull Market')
axes[2].fill_between(data.index, 0, regime_numeric, 
                     where=(regime_numeric < 0), color='red', alpha=0.3, label='Bear Market')
axes[2].set_title('Timeline des Régimes', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Régime')
axes[2].set_xlabel('Date')
axes[2].set_ylim(-1.5, 1.5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hmm_regime_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGraphique sauvegardé dans 'hmm_regime_detection.png'")


print("\n" + "="*70)
print("6. ANALYSE DES TRANSITIONS ENTRE RÉGIMES")
print("-"*70)

transition_matrix = best_model.transmat_

print("\nMatrice de Transition:")
print(f"{'':10s} {'→ Bull':>12s} {'→ Bear':>12s}")
print("-" * 35)
for i, from_label in enumerate(['Bull', 'Bear']):
    from_state = bull_state if i == 0 else bear_state
    to_bull = transition_matrix[int(from_state), int(bull_state)]
    to_bear = transition_matrix[int(from_state), int(bear_state)]
    print(f"{from_label+':':10s} {to_bull:>12.2%} {to_bear:>12.2%}")


bull_duration = 1 / (1 - transition_matrix[int(bull_state), int(bull_state)])
bear_duration = 1 / (1 - transition_matrix[int(bear_state), int(bear_state)])

print(f"\nDurée moyenne des régimes:")
print(f"  Bull Market: {bull_duration:.1f} jours (~{bull_duration/21:.1f} mois)")
print(f"  Bear Market: {bear_duration:.1f} jours (~{bear_duration/21:.1f} mois)")


print("\n" + "="*70)
print("7. INTERPRÉTATION FINANCIÈRE")
print("-"*70)

print("""
CARACTÉRISTIQUES DES RÉGIMES DÉTECTÉS:

BULL MARKET (Marché Haussier):
  • Rendements moyens positifs
  • Volatilité généralement plus faible
  • Confiance des investisseurs élevée
  • Tendance haussière des prix
  • Période de croissance économique

BEAR MARKET (Marché Baissier):
  • Rendements moyens négatifs
  • Volatilité généralement plus élevée
  • Incertitude et peur sur les marchés
  • Tendance baissière des prix
  • Possible récession ou ralentissement économique

APPLICATIONS PRATIQUES:
  1. Gestion de portefeuille adaptative
  2. Stratégies de couverture (hedging)
  3. Timing d'entrée/sortie du marché
  4. Allocation d'actifs dynamique
  5. Gestion des risques

LIMITES DU MODÈLE:
  • Détection a posteriori (pas de prédiction en temps réel)
  • Sensible aux paramètres initiaux
  • Suppose des distributions gaussiennes
  • Ne capture pas les événements extrêmes (black swans)
""")


regime_changes = (data['Regime'].diff() != 0).sum()
print(f"\nNombre de changements de régime détectés: {regime_changes}")
print(f"Fréquence moyenne: 1 changement tous les {len(data)/regime_changes:.0f} jours")


print("\n" + "="*70)
print("8. EXPORT DES RÉSULTATS")
print("-"*70)


results = data[['Close', 'Returns', 'Regime', 'Regime_Label']].copy()
results.to_csv('hmm_results.csv')
print("✓ Résultats sauvegardés dans 'hmm_results.csv'")

yearly_regimes = data.groupby([data.index.year, 'Regime_Label']).size().unstack(fill_value=0)
yearly_regimes['Bull_Pct'] = yearly_regimes['Bull'] / (yearly_regimes['Bull'] + yearly_regimes['Bear']) * 100

print("\nRépartition des régimes par année:")
print(yearly_regimes)

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"""
✓ Modèle HMM Gaussien avec {n_states} états ajusté avec succès

RÉSUMÉ:
  • {regime_stats_df.iloc[0]['Count']} jours en Bull Market ({regime_stats_df.iloc[0]['Percentage']:.1f}%)
  • {regime_stats_df.iloc[1]['Count']} jours en Bear Market ({regime_stats_df.iloc[1]['Percentage']:.1f}%)
  • {regime_changes} transitions de régime détectées
  • Rendement annualisé Bull: {regime_stats_df.iloc[0]['Mean']*252*100:.2f}%
  • Rendement annualisé Bear: {regime_stats_df.iloc[1]['Mean']*252*100:.2f}%

FICHIERS GÉNÉRÉS:
  • hmm_data_exploration.png - Exploration des données
  • hmm_regime_detection.png - Visualisation des régimes
  • hmm_results.csv - Données avec régimes identifiés
""")