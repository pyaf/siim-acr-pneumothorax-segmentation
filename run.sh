#python submission.py -f configs/288_senet154_f1_test.yaml
python submission.py -f configs/298_unetresnet34_f1_test.yaml -p train
python submission.py -f configs/019_unetdn121_f1_test.yaml -e 30 -p train
python submission.py -f configs/019_inresv2_f1_test.yaml -e 30 -p train
