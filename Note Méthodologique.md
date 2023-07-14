## Note méthodologique

###### Détails et description des méthodes employées au cours du projet, étape par étape.



### Entraînement du modèle :

Dans l'objectif de répondre à la problématique de notre projet qui consistait à prédire si , pour un client avec des informations données,
on accepte ou non sa demande de prêt, il nous fallait évidemment utiliser un modèle de machine learning. Ce modèle sera en effet capable d'apprendre des données
pré-existantes, et d'utiliser ces informations pour prédire la probabilité qu'un nouveau client rembourse ou non son prêt, et d'ainsi accepter ou non sa demande.

Pour entraîner le modèle, j'avais accès à une base de données, contenant pour chaque client qui fait une demande de prêt, des informations personnelles, sur sa vie et sa situation, ainsi que des données historiques sur ses éventuelles précédentes demande de prêt. Ces données était réparties sur plusieurs tables, liées entre elles notamment par l'identifiant du prêt. Afin d'entraîner le modèle il fallait donc commencer par traiter ces données afin de les regrouper en un seul dataset contenant un maximum d'informations sur chaque client. J'ai pour cela récupéré un feature engineering existant comme conseillé dans le descriptif du projet, qui traite chaque table individuellement en appliquant de nombreuses aggrégations afin de regrouper chaque demande de prêt sur une seule ligne. On obtient donc ainsi de nombreuses nouvelles colonnes, avec des données synthétisées à partir des données existantes. On rejoint enfin toutes ces nouvelles tables ensemble afin d'obtenir un dataset unique, contenant toutes les informations pour chaque client et permettant un entraînement optimal du modèle.

Il fallait ensuite évidemment choisir le meilleur modèle possible, et après avoir essayé différents algorithme afin de comparer leurs résultats, j'ai choisir d'utiliser LightGBM, qui était également le modèle le plus utilisé dans la majorité des notebooks sur la compétition Kaggle que j'ai pu observer. 

Lors de la cross validation, j'ai uniquement pris en compte un échantillon de ce dataset afin de réduire les temps de traitement, mais le modèle optimisé est au final évidemment entraîner sur l'ensemble des données d'entraînement.

  

### Déséquilibre des classes

Un des plus gros problèmes de notre dataset pour ce projet était le déséquilibre très importants des classes. En effet, il y avait évidemment beaucoup plus de prêts acceptés que réfusés, et il a donc fallu prendre ceci en compte dans l'entraînement de notre modèle. 
Pour cela, j'ai mis en place et comparer deux méthodes : Tout d'abord, en utilisant la librairie SMOTE pour oversampler la classe minoritaire, en créant de nouveaux individus proches des individus existants pour égaliser le ratio accepté/refusé. On peut utiliser cette méthode directement au sein de la cross-validation car la librairie imb-learn met à disposition une fonction Pipeline() qui permet de réaliser l'oversampling sur le set d'entraînement mais de conserver uniquement les individus existants pour la validation, ce qui nous permet également de passer des paramètres de SMOTE directement dans la cross-validation.

L'autre méthode que j'ai utilisée est beaucoup plus simple et fait uniquement appel à un paramètre présent notamment dans les modèles de régression linéaires et lightGBM, qui permet de préciser directement au modèle que les classes sont déséquilibrées, afin qu'il ajuste directement les poids attribués à chacune lors de l'apprentissage, ce qui se retrouve ensuite dans les prédictions.

Ces deux approches permettent donc d'adresser efficacement le problème de déséquilibre des classes, avec des résultats cependant  bien meilleurs pour la seconde, et c'est donc ce modèle sans SMOTE que j'ai utilisé par la suite.



### Coût métier, optimisation et métrique d'évaluation

Après avoir choisi le modèle, j'ai procédé à l'optimisation de ses paramètres. Pour cela, j'ai mis en place un nouveau scoreur pour évaluer plus précisément le coût métier, que j'ai utilisé dans les différentes étapes de cross validation pour optimiser les paramètres.
Ce scoreur a été créée dans l'optique de réduire au maximum les pertes potentielles lors de l'attribution des prêts. Il y a deux possibilités qui peuvent nous faire perdre de l'argent lors de cette décision : Refuser un client qui aurait remboursé sans problème, ce qui crée un manque à gagner, et Accepter un client qui ne va pas rembourser, ce qui engendre une perte nette. Cependant, ces deux cas ne sont pas égaux, il y a en effet plus d'argent à perdre dans le second que dans le premier. C'est donc cela que l'on va refléter dans ce scoreur, en créant une fonction simple qui prend en entrée les prédictions du modèles et les classes actuelles des individus, et qui crée ensuite un compteur d'erreur, qui augmente de 1 en cas de bon client refusé et de 10 en cas de mauvais client accepté. On a ainsi une importance x10 sur les mauvais payeurs acceptés, ce qui va diminuer la tolérance de notre modèle sur les clients à la limite. Ce score est cependant très dépendant de la taille de l'échantillon, donc j'ai pris soin de conserver la même taille d'échantillon partout afin de ne pas biaiser les résultats.

Concernant la cross-validation, j'ai utilisé 2 algorithmes : Tout d'abord le RandomSearchCV, avec un grand nombre d'itérations afin de balayer sur une plage de valeurs large, puis, en prenant un certain intervalle autour des valeurs trouvées avec la première méthode, j'ai procédé à un GridSearchCV pour trouver les meilleurs paramètres localement pour mon algorithme. L'optimisation n'a donc pas été faite sur l'AUC contrairement à l'objectif de la compétition, mais j'ai évidemment continuer d'observer ce score durant les différentes étapes afin de vérifier que ma méthode et mes résultats étaient tout de même bons. En vérification après l'optimisation, j'ai donc eu un score AUC de 0.68, ce qui est en dessous des scores optimaux trouvés pour la compétition, mais reste très satisfaisant pour notre projet en sachant que ce n'était pas notre critère d'optimisation.

La dernière étape concernait l'optimisation du seuil pour la prédiction. En effet, pour notre modèle de base, le seuil de prédiction, c'est à dire la probabilité à dépasser pour que le modèle prédisse 1 est de 0.5. On pouvait donc se demander s'il était possible d'obtenir de meilleurs résultats en faisant varier ce seuil. Cependant, avec les paramètres optimisés que l'on a trouvé, ce seuil de 0.5  reste le plus optimal en terme de score métier. Cela paraît normal car le modèle à été optimisé sur le scorer avec ce seuil à 0.5, mais il était compliqué de faire intervenir les différents seuils directement dans la cross-validation donc j'ai préféré garder ce seuil.*



### Synthèses des résultats 

Voici donc les résultats obtenus sur le set de validation mis à part lors de la création de notre échantillon d'apprentissage, contenant 3844 individus :

![image](https://github.com/Haelios/Projet_7_OC/assets/133202042/0f422995-dd60-49d7-b054-8dc756157d1f)

On peut voir ainsi avec la matrice de confusion les erreurs commises par le modèle, avec notamment en majorité des clients qui sont refusés alors qu'ils auraient remboursé, ce qui découle de notre score métier, qui pousse le modèle à être assez strict pour réduire les pertes au maximum. On a tout de même un AUC de 0.70 sur cet échantillon ce qui est très correct.

Et enfin les résultats obtenus pour l'ensemble des individus du set d'entraînement original moins ceux utilisés pour fit le modèle :
![image](https://github.com/Haelios/Projet_7_OC/assets/133202042/8019f0a7-1dd7-4b6b-959a-9d05571ebdab)

Evidemment le score métier s'envole car la population est démultipliée, mais on retrouve les mêmes tendances sur les erreurs commises, et toujours un AUC correct avec 0.684.


### Interprétabilité locale et globale

Pour étudier les spécifités de notre modèle, j'ai fait appel à la librairie SHAP qui créée un explainer du modèle et qui permet ensuite d'afficher les shap values. Ces valeurs sont calculées en étudiant l'influence de chaque variable sur la prédiction finale, et permettent ainsi d'afficher pour l'ensemble du modèle, mais également sur un individu à la fois, l'importance respective de chaque feature sur nos prédictions.
Voici donc le résumé de la feature importance de notre modèle, en gardant ici uniquement les 10 variables avec les valeurs les plus élevées, affichée grâce aux fonctions plots itnégrées à la librarie :
![image](https://github.com/Haelios/Projet_7_OC/assets/133202042/160e46b3-4ced-4057-a4be-70bcc19931ff)

On peut donc voir que les variables les plus importantes sont les 3 EXT_SOURCE de loin, suivies par des informations personnelles diverses.
Ensuite, on peut aller voir pour chaque individu les features importances locales, c'est à dire les variables qui ont eu la plus grande importance pour ce sujet en particulier, de cette façon :
![image](https://github.com/Haelios/Projet_7_OC/assets/133202042/6a2666dc-2f22-49c2-885b-7e2359858ed4)

Evidemment on retrouvera ici majoritairement les mêmes features que globalement car elles c'est leur importance locale pour tous les individus qui fait qu'elles le sont globalement, mais cela nous permet d'analyser pour chaque client quels sont les éventuels points faibles et forts de son dossier, afin qu'en obtenant ses résultats il puisse comprendre les raisons de ceux-ci et les points forts et faibles dans sa demande de prêt.


### Limites et possibles améliorations

J'ai évidemment dans ce projet rencontré des problèmes que j'ai du contourner, dont certains que j'ai déjà mentionné dans cette note. Cela a évidemment impacté les performances et avec plus de temps ou de moyens matériels il serait certainement possible d'améliorer nos résultats. La première chose et la plus évidente est évidemment le feature engineering, que j'ai repris d'un notebook existant, qui est plutôt complète et qui m'a permis d'avancer rapidement sur la suite sans trop m'épancher dessus, mais il est certainement possible d'obtenir quelque chose d'encore meilleur à ce niveau. Ensuite, comme je l'ai précisé plus tôt, j'ai du procéder à l'optimisation de mon modèle sur un échantillon du dataset car les temps de traitement pour la cross validation étaient bien trop élevés sinon, et on pourrait donc sûrement améliorer notre modèle en traitant directement tout le dataset à l'aide d'un peu plus de puissance de calcul. D'autre part, le calcul du score métier ne prends en compte que les pertes potentielles, mais il me paraîtrait également pertinent d'ajouter à cela le gain engendré par un prêt accordé à un bon client, ce qui pourrait pousser le modèle à valider plus de prêts malgré les risques. Encore faudrait-il pour cela évaluer le coefficient d'un tel gain, comme on l'a fait avec le ratio 1:10 pour les pertes.
Enfin, une limite qui me paraît importante notamment dans la transmission des données aux clients, est le fait que l'on possède quasiment aucune information sur ce que représentes ces EXT_SOURCE, alors qu'elles sont les variables les plus importantes pour notre modèle et donc les plus importantes dans la décision finale. Il serait donc intéressant d'avoir plus d'informations sur cela notamment pour transmettre au client par la suite.



### Analyse du Data Drift




