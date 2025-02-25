#!/bin/bash


python finetune_lm.py --dataset cadec --epochs 20 --batch_size 16 --pt_name dmis-lab/biobert-base-cased-v1.2
python finetune_lm.py --dataset cadec --epochs 20 --batch_size 16 --pt_name emilyalsentzer/Bio_ClinicalBERT
python finetune_lm.py --dataset cadec --epochs 20 --batch_size 16 --pt_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
python finetune_lm.py --dataset cadec --epochs 20 --batch_size 16 --pt_name Lianglab/PharmBERT-uncased

python finetune_lm.py --dataset share13 --epochs 20 --batch_size 6 --pt_name emilyalsentzer/Bio_ClinicalBERT
python finetune_lm.py --dataset share13 --epochs 20 --batch_size 6 --pt_name dmis-lab/biobert-base-cased-v1.2
python finetune_lm.py --dataset share13 --epochs 20 --batch_size 6 --pt_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
python finetune_lm.py --dataset share13 --epochs 20 --batch_size 6 --pt_name Lianglab/PharmBERT-uncased

python finetune_lm.py --dataset share14 --epochs 20 --batch_size 6 --pt_name emilyalsentzer/Bio_ClinicalBERT
python finetune_lm.py --dataset share14 --epochs 20 --batch_size 6 --pt_name Lianglab/PharmBERT-uncased
python finetune_lm.py --dataset share14 --epochs 20 --batch_size 6 --pt_name dmis-lab/biobert-base-cased-v1.2
python finetune_lm.py --dataset share14 --epochs 20 --batch_size 6 --pt_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract