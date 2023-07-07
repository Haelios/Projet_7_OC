## Note méthodologique

###### Détails et description des méthodes employées au cours du projet, étape par étape.


### Entraînement du modèle :


Dans l'objectif de répondre à la problématique de notre projet qui consistait à prédire si , pour un client avec des informations données,
on accepte ou non sa demande de prêt, il nous fallait évidemment utiliser un modèle de machine learning. Ce modèle sera en effet capable d'apprendre des données
pré-existantes, et d'utiliser ces informations pour prédire la probabilité qu'un nouveau client rembourse ou non son prêt, et d'ainsi accepter ou non sa demande.
  
Pour répondre à ce problème, de nombreux modèles existent, plus ou moins sophistiqués. J'ai donc commencé par procéder à des essais de divers modèles, sans modifier leurs hyperparamètres, afin de les comparer et de conserver uniquement le plus preformant. Pour cette première évaluation, j'ai comparé les modèles à l'aide de l'AUC car c'est la métrique utilisée pour évaluer les résultats soumis dans la compétition Kaggle. J'ai utilisé comme baseline un DummyClassifier, modèle le plus simple possible, qui se contente de **************. J'ai ensuite essayé un modèle de regression linéaire, qui offre déjà de meilleurs résultats que le modèle de base, mais toujours améliorable. Enfin, j'ai utilisé LightGBM, qui est un modèle de Boosting très performant, ainsi que celui qui revenait majoritairement dans les divers Notebook en relation avec la relation Kaggle que j'ai pu observer. C'est donc ce modèle qui m'a permis d'obtenir les meilleurs résultats de base, avant même de toucher à ses hyperparamètres, et c'est celui que j'ai gardé par la suite.
