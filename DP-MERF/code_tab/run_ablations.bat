@echo off
echo Starting ablations...

python run_ablations_tabular.py --dataset adult --sweep B1 --baseline
python run_ablations_tabular.py --dataset adult --sweep B5 --baseline
python run_ablations_tabular.py --dataset adult --sweep B10 --baseline

rem After reviewing B results, patch with best k_splits
python run_ablations_tabular.py --dataset adult --sweep A --best_k_splits 25 --baseline
python run_ablations_tabular.py --dataset adult --sweep C --best_k_splits 25 --best_vote_rounds 5 --baseline

echo All done!
pause