# FLAIN: Mitigating Backdoor Attacks in Federated Learning via Flipping Weight Updates of Low-Activation Input Neurons

```bash
python src/federated.py --data=mnist --local_ep=2 --bs=256 --data_distribution=non_iid --num_agents=50 --train_rounds=200 --backdoor_frac=0.2  --poison_frac=0.5
```
# Note that when applying FLAIN for defense, you need to set the model save path in <aggregation.py>. 
# During the defense process for each dataset, these saved models should be loaded accordingly. 
# Additionally, under the non-IID setting, the non-IID split configuration should be stored in the corresponding file within <utils.py> (i.e., the Non-IID file).



