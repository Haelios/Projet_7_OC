# Projet_7_OC

### Dossier GitHub comprenant les différents éléments réalisés dans le cadre de mon Projet 7 du parcours Data Scientist d'OpenClassrooms.

Ce projet ce base sur le dataset de la compétition Kaggle disponible [ici](https://www.kaggle.com/competitions/home-credit-default-risk/overview). Le but initial de cette compétiion étant de mettre en place un modèle de classification permettant de prédire si un client va être ou non en mesure ce rembourser un prêt dont il fait la demande, et donc de décider d'accorder ou non le prêt selon la prédiction.

Pour résoudre ce problème on a accès à une database contenant de nombreuses informations sur chaque client unique identifié par un ID_CLIENT. La première étape sera évidemment de traiter ces données brutes afin d'en obtenir le plus d'informations intéressantes pour entraîner notre modèle. Comme recommandé dans l'énoncé du projet, j'ai décidé de prendre un feature engineering préfait par un participant à la compétition Kaggle afin de faciliter le travail. J'ai évidemment appliqué quelques modifications à ce script afin d'en faciliter la compréhension et l'utilisation.

Après avoir traité les données, on va commencer par une étude afin de trouver le meilleur modèle à utiliser pour ce projet. Pour cela on va mettre en place différents algorithmes de classification afin de comparer leurs performances, et une fois qu'on aura sélectionné le plus performant on pourra appliquer différentes fonctions de cross validation afin d'optimiser les hyperparamètres et d'obtenir les meilleurs résultats possibles avec notre modèle. Toutes ces études sont à retrouver au sein du notebook jupyter sur le GitHub.
