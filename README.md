# 📊 GridGambler
---

---

## 1️⃣ MODELE SYSTEMOWE (Techniczne)  
**Cel:** Optymalizacja efektywności elektrociepłowni, przewidywanie strat, analiza systemu pod kątem decyzji Tradera  

| Model | Opis | Technologie / Algorytmy | Biblioteki |
|-------|------|-------------------------|------------|
| **Model zapotrzebowania na ciepło** | Prognozowanie zużycia ciepła w różnych częściach miasta. Uwzględnia temperaturę, dzień tygodnia, godziny szczytu. Kluczowe dla Tradera – mówi, czy można obniżyć produkcję bez ryzyka. | LSTM, Prophet, ARIMA, XGBoost | [statsmodels](https://www.statsmodels.org/), [prophet](https://facebook.github.io/prophet/), [tensorflow](https://www.tensorflow.org/) |
| **Model strat cieplnych** | Szacowanie strat w sieci ciepłowniczej. Kluczowe dla Tradera – jeśli mamy duże straty, nie opłaca się podbijać temperatury. | Modele fizyczne (równania strat ciepła), Graph Neural Networks (GNN) | [networkx](https://networkx.github.io/), [PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/) |
| **Model efektywności spalania** | Analiza efektywności spalania – czy możemy spalić więcej/mniej, jaką mieszankę zastosować? Informuje operatora i Tradera o możliwych strategiach. | Regresja, modele spalania, Monte Carlo | [XGBoost](https://xgboost.readthedocs.io/), [OpenFOAM](https://openfoam.org/), [SHAP](https://shap.readthedocs.io/) |
| **Model kosztów operacyjnych** | Optymalizacja kosztów: czy lepiej spalać odpady, gaz, węgiel? Daje Traderowi dane do wyboru strategii. | Constraint Optimization, Bayesian Optimization | [cvxpy](https://www.cvxpy.org/), [bayesian-optimization](https://github.com/bayesian-optimization/BayesianOptimization) |
| **Model awaryjności sieci** | Prognozowanie awarii na podstawie danych historycznych. Przydatne dla Tradera – jeśli ryzyko awarii rośnie, może lepiej ograniczyć agresywną strategię. | Time-Series Anomaly Detection | [isolation-forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [autoencoders](https://keras.io/examples/) |

---

## 2️⃣ MODELE RYNKOWE (Predykcja & Handel Energią)  
**Cel:** Przewidywanie cen, zmienności rynku, decyzje o strategii  

| Model | Opis | Technologie / Algorytmy | Biblioteki |
|-------|------|-------------------------|------------|
| **Model prognozy cen energii** | Przewiduje ceny energii cieplnej i elektrycznej, co pozwala Traderowi dostosować strategię. | Time-Series Forecasting (Transformers, XGBoost, LSTM) | [sklearn](https://scikit-learn.org/), [tensorflow](https://www.tensorflow.org/), [pytorch](https://pytorch.org/) |
| **Model wpływu wydarzeń rynkowych** | Analizuje wiadomości i politykę, by określić ich wpływ na ceny. Trader wie, czy np. czekają nas zmiany w podatkach CO₂. | NLP + Sentiment Analysis (BERT, FinBERT) | [transformers](https://huggingface.co/transformers/), [nltk](https://www.nltk.org/), [spacy](https://spacy.io/) |
| **Model optymalizacji zakupów paliwa** | Ocenia, kiedy opłaca się kupić paliwo, aby było najtaniej. Trader dostaje dane o cenach, ale to nie on podejmuje decyzje zakupowe. | Game Theory + Optimization | [cvxpy](https://www.cvxpy.org/), [pulp](https://coin-or.github.io/pulp/) |
| **Model zmienności popytu** | Przewiduje anomalie w zapotrzebowaniu na energię (np. nagły atak zimy). | Hidden Markov Models, Gaussian Processes | [hmmlearn](https://hmmlearn.readthedocs.io/), [GPy](https://sheffieldml.github.io/GPy/) |
| **Model wpływu polityki klimatycznej** | Estymuje wpływ przyszłych regulacji na ceny energii. | Bayesian Inference, RL | [pymc3](https://docs.pymc.io/), [stable-baselines3](https://stable-baselines3.readthedocs.io/) |

---

## 3️⃣ MODELE STRATEGICZNE (Trader/Gambler)  
**Cel:** Maksymalizacja zysku przez agresywne strategie zarządzania produkcją i sprzedażą  

| Model | Opis | Technologie / Algorytmy | Biblioteki |
|-------|------|-------------------------|------------|
| **Trader AI – Model strategii spekulacyjnych** | Kluczowy model decydujący, kiedy grać agresywnie, a kiedy zachować ostrożność. Dostaje dane z modeli systemowych i rynkowych. | Reinforcement Learning (DQN, PPO), Monte Carlo | [stable-baselines3](https://stable-baselines3.readthedocs.io/), [ray.rllib](https://docs.ray.io/en/latest/rllib.html) |
| **Model optymalizacji pracy kotłów w ekstremach** | Analizuje, czy warto zwiększyć produkcję lub przydusić kocioł dla maksymalizacji zysku. | Constraint Optimization, RL | [cvxpy](https://www.cvxpy.org/), [gurobi](https://www.gurobi.com/) |
| **Model analizy zachowań rynkowych** | Przewiduje decyzje innych uczestników rynku. Czy konkurenci podbiją ceny? | Game Theory (Nash Equilibrium, Minimax) | [nashpy](https://github.com/drvinceknight/Nashpy), [gambit](https://www.gambit-project.org/) |
| **Model zarządzania mocą szczytową** | Przewiduje, kiedy najlepiej dostarczyć ciepło, aby uzyskać najlepsze ceny lub warunki rynkowe. | RL, Multi-Agent Learning | [maddpg](https://github.com/openai/maddpg), [pettingzoo](https://www.pettingzoo.ml/) |
| **Model elastycznego wykorzystania bezwładności cieplnej** | Określa, czy możemy manipulować bezwładnością cieplną, aby zwiększyć zyski. | RL, Dynamic Programming | [gym](https://gym.openai.com/), [jax](https://jax.readthedocs.io/) |

---

## 4️⃣ META-TESTER – Pętla Ulepszania  

| Moduł | Opis | Technologie / Algorytmy | Biblioteki |
|-------|------|-------------------------|------------|
| **Meta-Tester AI** | Analizuje decyzje Tradera i porównuje je z rzeczywistością. Jeśli Trader podjął złą decyzję, model poprawia strategię. | Modelowanie kontrfaktyczne, Simulacja Monte Carlo | [causalml](https://github.com/uber/causalml), [shap](https://shap.readthedocs.io/) |
