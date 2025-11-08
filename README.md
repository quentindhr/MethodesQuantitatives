# Méthodes Quantitatives - Analyse de Séries Temporelles

Ce projet contient trois exercices pratiques sur l'analyse de séries temporelles appliquée à des données financières et économiques. Chaque exercice couvre une méthode différente : décomposition de séries temporelles, modélisation SARIMA, et détection de régimes avec les modèles de Markov cachés (HMM).

## 📋 Table des matières

- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Exercices](#exercices)
  - [Exercice 1 : Décomposition de Séries Temporelles](#exercice-1--décomposition-de-séries-temporelles)
  - [Exercice 2 : Modélisation SARIMA](#exercice-2--modélisation-sarima)
  - [Exercice 3 : HMM pour la Détection de Régimes](#exercice-3--hmm-pour-la-détection-de-régimes)
- [Où Trouver les Réponses](#où-trouver-les-réponses)
- [Résultats et Visualisations](#résultats-et-visualisations)
- [Références](#références)

---

## 📁 Structure du Projet

```
MethodesQuantitatives/
│
├── README.md                                    # Ce fichier
│
├── dataset1.txt                                 # Données ventes retail (1992-2025)
├── dataset2.txt                                 # Données consommation électrique (2015-2024)
├── dataset3.csv                                 # Données S&P 500 (2015-2024)
│
├── exercise1_timeseries_decomposition.py        # Exercice 1 - Décomposition
├── exercise2_sarima_model.py                    # Exercice 2 - SARIMA
├── exercise3_hmm_regime_detection.py            # Exercice 3 - HMM
├── generate_dataset3.py                         # Script pour générer dataset3.csv
│
└── outputs/                                     # Dossier pour les résultats
    ├── decomposition.png
    ├── sarima_acf_pacf.png
    ├── sarima_residuals.png
    ├── sarima_forecast.png
    ├── hmm_data_exploration.png
    ├── hmm_regime_detection.png
    └── hmm_results.csv
```

---

## 🔧 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le repository** (si vous ne l'avez pas déjà fait) :
```bash
git clone https://github.com/quentindhr/MethodesQuantitatives.git
cd MethodesQuantitatives
```

2. **Créer un environnement virtuel** (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate     # Sur Windows
```

3. **Installer les dépendances** :
```bash
pip install pandas numpy matplotlib scipy scikit-learn statsmodels hmmlearn openpyxl
```

### Liste des dépendances

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
statsmodels>=0.13.0
hmmlearn>=0.3.0
openpyxl>=3.0.0
```

---

## 📊 Exercices

### Exercice 1 : Décomposition de Séries Temporelles

**Objectif** : Décomposer une série temporelle de ventes retail en composantes de tendance, saisonnalité et résidus.

**Fichier** : `exercise1_timeseries_decomposition.py`  
**Données** : `dataset1.txt` (ventes mensuelles retail 1992-2025)

#### Comment exécuter

```bash
python exercise1_timeseries_decomposition.py
```

#### Ce que fait le code

1. **Charge les données** de ventes mensuelles retail
2. **Décompose la série** en trois composantes via régression linéaire :
   - **Tendance** : croissance à long terme
   - **Saisonnalité** : variations mensuelles récurrentes
   - **Résidus** : fluctuations aléatoires
3. **Analyse les résidus** :
   - Test de normalité (Shapiro-Wilk)
   - Autocorrélation pour détecter des patterns non capturés
4. **Génère des prévisions** pour 2025
5. **Évalue la qualité** du modèle (R², RMSE, MAE, MAPE)

#### Réponses aux questions de l'exercice

**Question 1** : Utilisation de régression pour séparer les composantes  
→ **Réponse** : Lignes 18-30 du code (régression avec variables temporelles et dummy saisonnières)

**Question 2** : Visualisation de chaque composante  
→ **Réponse** : Lignes 40-75 (4 graphiques : série originale, tendance, saisonnalité, résidus)

**Question 3** : Analyse de la qualité de décomposition  
→ **Réponse** : Lignes 78-100 (affiche dans la console) :
- Résidus non-normaux → décomposition imparfaite
- Autocorrélation élevée (0.98) → patterns non capturés
- Suggère l'utilisation de modèles plus complexes (SARIMA)

**Question 4** : Prévisions 2025 et qualité du modèle  
→ **Réponse** : Lignes 105-145 (affiche dans la console) :
- Comparaison prévisions vs valeurs réelles Jan-Août 2025
- Métriques : MAE, RMSE, MAPE pour quantifier la précision
- R² pour mesurer l'ajustement global

#### Sorties générées

- **Console** : Statistiques complètes et réponses aux questions
- **decomposition.png** : Visualisation des 4 composantes

---

### Exercice 2 : Modélisation SARIMA

**Objectif** : Modéliser et prévoir une série temporelle saisonnière avec SARIMA.

**Fichier** : `exercise2_sarima_model.py`  
**Données** : `dataset2.txt` (consommation électrique mensuelle 2015-2024)

#### Comment exécuter

```bash
python exercise2_sarima_model.py
```

#### Ce que fait le code

1. **Exploration des données** : chargement et inspection
2. **Analyse de stationnarité** :
   - Test Augmented Dickey-Fuller (ADF)
   - Différenciation si nécessaire (d et D)
3. **Identification des paramètres** :
   - Graphiques ACF/PACF pour déterminer (p, d, q)
   - Paramètres saisonniers (P, D, Q, s)
4. **Construction du modèle** :
   - Test de plusieurs configurations SARIMA
   - Sélection basée sur AIC/BIC
5. **Évaluation** :
   - Métriques : MAE, RMSE, MAPE
   - Analyse des résidus (normalité, autocorrélation)
6. **Prévisions** : génération de prévisions futures avec visualisation

#### Réponses aux questions de l'exercice

**Question 1** : Exploration des données  
→ **Réponse** : Lignes 13-45 (affiche dans la console les statistiques et détecte les valeurs manquantes)

**Question 2** : Analyse de stationnarité  
→ **Réponse** : Lignes 48-85 (affiche dans la console) :
- Test ADF avec p-values
- Recommandation pour d (différenciation ordinaire)
- Recommandation pour D (différenciation saisonnière)

**Question 3** : Identification des paramètres SARIMA  
→ **Réponse** : Lignes 88-130 :
- Graphiques ACF/PACF sauvegardés dans `sarima_acf_pacf.png`
- Interprétation des pics pour identifier p, q, P, Q
- Recommandations affichées dans la console

**Question 4** : Construction du modèle  
→ **Réponse** : Lignes 133-160 :
- Test de plusieurs configurations
- Sélection du meilleur modèle (AIC le plus bas)
- Résumé complet du modèle affiché

**Question 5** : Évaluation de la performance  
→ **Réponse** : Lignes 163-200 :
- MAE, RMSE, MAPE affichés dans la console
- Analyse des résidus (normalité, patterns restants)
- Graphiques de diagnostic dans `sarima_residuals.png`

**Question 6** : Prévisions et visualisation  
→ **Réponse** : Lignes 203-240 :
- Prévisions pour 12 mois futurs
- Graphique comparatif dans `sarima_forecast.png`
- Valeurs de prévisions affichées dans la console

#### Sorties générées

- **Console** : Tous les résultats, statistiques et interprétations
- **sarima_acf_pacf.png** : Graphiques ACF/PACF pour identification des paramètres
- **sarima_residuals.png** : Diagnostic des résidus
- **sarima_forecast.png** : Prévisions vs données réelles

---

### Exercice 3 : HMM pour la Détection de Régimes

**Objectif** : Détecter les régimes cachés (bull/bear markets) dans les données S&P 500 avec un modèle de Markov caché (HMM).

**Fichier** : `exercise3_hmm_regime_detection.py`  
**Données** : `dataset3.csv` (prix journaliers S&P 500 2015-2024)

#### Comment exécuter

1. **Générer les données** (si dataset3.csv n'existe pas) :
```bash
python generate_dataset3.py
```

2. **Exécuter l'exercice** :
```bash
python exercise3_hmm_regime_detection.py
```

#### Ce que fait le code

1. **Charge les données** S&P 500 et calcule les rendements journaliers
2. **Ajuste un HMM Gaussien** :
   - Test avec 2 et 3 états cachés
   - Sélection basée sur log-likelihood, AIC, BIC
3. **Identifie les régimes** :
   - Assignation de chaque jour à un régime (bull ou bear)
   - Classification basée sur les rendements moyens
4. **Analyse les caractéristiques** :
   - Rendements moyens par régime
   - Volatilité par régime
   - Interprétation financière
5. **Visualise** :
   - Prix colorés par régime
   - Rendements colorés par régime
   - Timeline des régimes
6. **Matrice de transition** : probabilités de passage entre régimes

#### Réponses aux questions de l'exercice

**Question 1** : Hypothèse des régimes cachés  
→ **Réponse** : Lignes 93-110 :
- HMM avec 2 états : bull market et bear market
- États influencent les rendements observés

**Question 2** : Ajustement du HMM Gaussien  
→ **Réponse** : Lignes 93-118 :
- Test de plusieurs nombres d'états (2 et 3)
- Sélection du modèle optimal
- Log-likelihood, AIC, BIC affichés

**Question 3** : Attribution des régimes et visualisation  
→ **Réponse** : Lignes 121-215 :
- Prédiction des états cachés pour chaque jour
- Visualisation avec coloration dans `hmm_regime_detection.png`
- Timeline claire des transitions de régimes

**Question 4** : Caractéristiques et interprétation financière  
→ **Réponse** : Lignes 140-175 (affiche dans la console) :
- **Bull Market** : rendements positifs, volatilité faible
- **Bear Market** : rendements négatifs, volatilité élevée
- Rendements annualisés et volatilité annualisée
- Pourcentage de temps dans chaque régime
- Interprétation économique complète (lignes 230-270)

#### Sorties générées

- **Console** : Statistiques complètes, caractéristiques des régimes, interprétation
- **hmm_data_exploration.png** : Prix et rendements
- **hmm_regime_detection.png** : Visualisation complète avec régimes colorés
- **hmm_results.csv** : Données exportées avec régimes identifiés

---

## 🎯 Où Trouver les Réponses

### Format des réponses

Toutes les réponses aux questions des exercices sont disponibles dans **DEUX formats** :

#### 1. Console / Terminal

Lorsque vous exécutez chaque script, **toutes les réponses sont affichées directement dans la console** avec :
- ✅ Des sections clairement identifiées
- 📊 Des statistiques et métriques
- 💡 Des interprétations et recommandations
- ⚠️ Des avertissements sur les limitations

**Exemple** pour l'Exercice 1 :
```
============================================================
RÉPONSES AUX QUESTIONS DE L'EXERCICE
============================================================

1. DÉCOMPOSITION (Tendance, Saisonnalité, Résidus):
   ✓ Effectuée avec succès par régression linéaire
   ✓ Visualisations créées dans 'decomposition.png'

2. ANALYSE DES RÉSIDUS:
   ✗ La décomposition n'est PAS totalement satisfaisante:
     - Résidus non normaux (test de Shapiro-Wilk rejeté)
     ...
```

#### 2. Fichiers graphiques

Chaque exercice génère des **visualisations PNG** qui répondent aux questions visuellement :

**Exercice 1** :
- `decomposition.png` → Questions 1 et 2 (décomposition et visualisation)

**Exercice 2** :
- `sarima_acf_pacf.png` → Question 3 (identification des paramètres)
- `sarima_residuals.png` → Question 5 (évaluation)
- `sarima_forecast.png` → Question 6 (prévisions)

**Exercice 3** :
- `hmm_data_exploration.png` → Exploration initiale
- `hmm_regime_detection.png` → Questions 3 et 4 (régimes et visualisation)

### Tableau récapitulatif

| Exercice | Question | Où trouver la réponse | Lignes de code |
|----------|----------|------------------------|----------------|
| **Ex1** | Q1 : Décomposition | Console + decomposition.png | 18-30 |
| **Ex1** | Q2 : Visualisation | decomposition.png | 40-75 |
| **Ex1** | Q3 : Qualité résidus | Console (section ANALYSE) | 78-100 |
| **Ex1** | Q4 : Prévisions 2025 | Console (section PRÉVISIONS) | 105-145 |
| **Ex2** | Q1 : Exploration | Console (section 1) | 13-45 |
| **Ex2** | Q2 : Stationnarité | Console (section 2) | 48-85 |
| **Ex2** | Q3 : Paramètres | Console + sarima_acf_pacf.png | 88-130 |
| **Ex2** | Q4 : Construction | Console (section 4) | 133-160 |
| **Ex2** | Q5 : Évaluation | Console + sarima_residuals.png | 163-200 |
| **Ex2** | Q6 : Prévisions | Console + sarima_forecast.png | 203-240 |
| **Ex3** | Q1 : Hypothèse régimes | Console (section 3-4) | 93-110 |
| **Ex3** | Q2 : Ajustement HMM | Console (section 3) | 93-118 |
| **Ex3** | Q3 : Visualisation | hmm_regime_detection.png | 121-215 |
| **Ex3** | Q4 : Interprétation | Console (section 4 et 7) | 140-270 |

---

## 📈 Résultats et Visualisations

### Exercice 1 - Décomposition

**Fichier généré** : `decomposition.png`

Contient 4 graphiques empilés :
1. Série temporelle originale (ventes retail)
2. Composante de tendance (croissance linéaire)
3. Composante saisonnière (variations mensuelles)
4. Résidus (fluctuations inexpliquées)

**Métriques clés** (affichées dans la console) :
- R² : qualité de l'ajustement
- RMSE : erreur quadratique moyenne
- MAE : erreur absolue moyenne
- MAPE : erreur en pourcentage

### Exercice 2 - SARIMA

**Fichiers générés** :
- `sarima_acf_pacf.png` : Identification des paramètres
- `sarima_residuals.png` : Diagnostic du modèle
- `sarima_forecast.png` : Prévisions futures

**Modèle recommandé** : SARIMA(p,d,q)(P,D,Q)₁₂
- p, q : ordre AR et MA
- d : différenciation
- P, Q : ordre saisonnier AR et MA
- D : différenciation saisonnière
- s=12 : période saisonnière (mensuelle)

### Exercice 3 - HMM

**Fichiers générés** :
- `hmm_data_exploration.png` : Prix et rendements S&P 500
- `hmm_regime_detection.png` : Régimes détectés (3 graphiques)
- `hmm_results.csv` : Données complètes avec régimes

**Régimes identifiés** :
- 🟢 **Bull Market** : rendements positifs, faible volatilité
- 🔴 **Bear Market** : rendements négatifs, haute volatilité

**Matrice de transition** : Probabilités de passer d'un régime à l'autre

---

## 📚 Concepts Clés

### Décomposition de Séries Temporelles
- **Tendance** : mouvement à long terme
- **Saisonnalité** : patterns récurrents (mensuel, annuel)
- **Résidus** : variations aléatoires

### SARIMA - Seasonal AutoRegressive Integrated Moving Average
- **AR (p)** : composante autoregressive
- **I (d)** : différenciation pour stationnarité
- **MA (q)** : composante moving average
- **Saisonnier (P,D,Q,s)** : capture les patterns saisonniers

### HMM - Hidden Markov Model
- **États cachés** : régimes non observés (bull/bear)
- **Observations** : rendements observés
- **Transitions** : probabilités de changer de régime
- **Émissions** : distributions des rendements par régime

---

## 🔍 Interprétation des Résultats

### Exercice 1

✅ **Bonne décomposition si** :
- Résidus proches de 0 en moyenne
- Résidus normalement distribués
- Faible autocorrélation des résidus
- R² > 0.90

⚠️ **Amélioration nécessaire si** :
- Autocorrélation élevée (>0.3)
- Résidus non-normaux
- Patterns visuels dans les résidus

### Exercice 2

✅ **Bon modèle SARIMA si** :
- Résidus non-autocorrélés (ACF proche de 0)
- Résidus normalement distribués (Q-Q plot linéaire)
- MAPE < 5% (bonnes prévisions)
- AIC/BIC minimisés

⚠️ **Amélioration nécessaire si** :
- Résidus autocorrélés → essayer d'autres ordres
- Erreurs de prévision élevées → ajouter variables exogènes

### Exercice 3

✅ **HMM valide si** :
- Régimes bien séparés (moyennes distinctes)
- Interprétation financière cohérente
- Durée des régimes réaliste (quelques mois)
- Nombre de transitions raisonnable

⚠️ **Limitation** :
- Détection a posteriori (pas de prédiction temps réel)
- Sensible aux paramètres initiaux

---

## 🚀 Commandes Rapides

```bash
# Exécuter tous les exercices
python exercise1_timeseries_decomposition.py
python exercise2_sarima_model.py
python generate_dataset3.py  # Si nécessaire
python exercise3_hmm_regime_detection.py

# Nettoyer les sorties
rm -f *.png *.csv

# Réinstaller les dépendances
pip install -r requirements.txt
```

---

## 👨‍💻 Auteurs

**Quentin Deharo** (@quentindhr)
**Cornel Cristea** (@scornel09)

## 📝 License

Ce projet est à des fins éducatives dans le cadre du cours de Méthodes Quantitatives.
