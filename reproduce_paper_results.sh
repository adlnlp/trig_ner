#!/bin/bash


python inference.py --config ./config/cadec.json --session_name CADEC_Overall_Inference --checkpoint ./output/best_results/CADEC_best_overall.pt --predict_test
python inference.py --config ./config/cadec_best_discent.json --session_name CADEC_DiscEnt_Inference --checkpoint ./output/best_results/CADEC_best_discent.pt --predict_test

python inference.py --config ./config/share13.json --session_name SHARE13_Overall_Inference --checkpoint ./output/best_results/SHARE13_best_overall.pt --predict_test
python inference.py --config ./config/share13_best_discent.json --session_name SHARE13_DiscEnt_Inference --checkpoint ./output/best_results/SHARE13_best_discent.pt --predict_test

python inference.py --config ./config/share14.json --session_name SHARE14_Overall_Inference --checkpoint ./output/best_results/SHARE14_best_overall.pt --predict_test
python inference.py --config ./config/share14_best_discent.json --session_name SHARE14_DiscEnt_Inference --checkpoint ./output/best_results/SHARE14_best_discent.pt --predict_test