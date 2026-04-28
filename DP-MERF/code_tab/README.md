# Tabular data experiments

To run the tabular experiments it is enough to run the script single_generator_priv_all.py with some of the arguments.

## Datasets

We include in the supplementary materials to the links of each dataset. We include also all the datasets in this dataset, except for the credit dataset which is too large. Intrusion (and also adult) datasets can downloaded with the SDGym package.

## Arguments

The arguments are following:  

dataset - type of dataset, choices: epileptic, credit, census, cervical, adult, isolet, intrusion, covertype, default=adult  
private - if running the generator with privacy guarantees, choices: 0, (non-private), 1 (private), by default non-private training is run  
epochs - number of training epochs for generator  
batch - batch size  
num_features - number of features  
undersample - the undersampling to the most prelevant class label  
repeat - number of repetitions of the entire experiment  
classifiers - a list of methods to test, by default all methods are run.  

0 - LogisticRegression  
1 - GaussianNB  
2 - BernoulliNB  
3 - LinearSVC  
4 - DecisionTreeClassifier  
5 - LinearDiscriminantAnalysis  
6 - AdaBoostClassifier  
7 - BaggingClassifier  
8 - RandomForestClassifier  
9 - GradientBoostingClassifier  
10 - MLP  
11 - XGBoost  

## Example

-  `python single_generator_priv_all.py --dataset credit --private 0 --epochs 2000 --batch 0.3 --num_features 1000 --undersample 0.5 --repeat 2 --classifiers 1 2 5`


# Step 1 — find best k_splits on adult
python run_ablations_tabular.py --dataset adult --sweep B1

# Step 2 — find best vote_rounds
python run_ablations_tabular.py --dataset adult --sweep A --best_k_splits 25 --baseline

# Step 3 — find best oversample_factor
python run_ablations_tabular.py --dataset adult --sweep C --best_k_splits 25 --best_vote_rounds 1

# Step 4 — find best generator_fraction
python run_ablations_tabular.py --dataset adult --sweep D --best_k_splits 25 --best_vote_rounds 1 --best_oversample 0.5

# Step 5 — test num_synth_factor
python run_ablations_tabular.py --dataset adult --sweep E --best_k_splits 25 --best_vote_rounds 1 --best_oversample 0.5

# Run on a different dataset with the same best config
python run_ablations_tabular.py --dataset census --sweep B


python single_generator_priv_all.py --dataset adult --skip_vote \
    --epochs 200 --batch 0.1 --undersample 0.4 \
    --num_features 2000 --repeat 3