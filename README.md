This code aims at simulating various attack and defense mechanism in fl.

### Depdendencies (tentative)
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1

### Data Preparation
---
1. For Southwest Airline (for CIFAR-10) and traditional Cretan costumes (for ImageNet) edge-case example, most of the collected edge-case datasets are available in `./saved_datasets`. 

### Running Experients:
---
The main script is `./main_averaging.py`, to launch the jobs, we provide a script `./run_main.sh`. And we provide a detailed description on the arguments.


| Argument                      | Description                                                                                            |
| ----------------------------- |--------------------------------------------------------------------------------------------------------|
| `fraction` | Only used for EMNIST, varying the fraction of poisioned data points in attacker's poisoned dataset.    |
| `lr` | Inital learning rate that will be used for local training process.                                     |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam.                                                        |
| `dataset`      | Dataset to use.                                                                                        |
| `model`      | Model to use.                                                                                          |
| `gamma` | the factor of learning rate decay, i.e. the effective learning rate is `lr*gamma^t`.                   |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam.                                                        |
| `num_nets` | The total number of available users e.g. 3383 for EMNIST and 200 for CIFAR-10.                         |
| `fl_round` | maximum number of FL rounds for the code to run.                                                       |
| `part_nets_per_round` | Number of active users that are sampled per FL round to participate.                                   |
| `local_train_period` | Number of local training epochs that the honest users can run.                                         |
| `adversarial_local_training_period`  | Number of local training epochs that the attacker(s) can run.                                          |
| `fl_mode`    | `fixed-freq` or `fixed-pool` for fixed frequency and fixed pool attacking settings.                    |
| `attacker_pool_size`    | Number of attackers in the total number of available users, used only when `fl_mode=fixed-pool`.       |
| `defense_method`    | Defense method over the data center end.  Types: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa|soft_hard                                                      |
| `stddev` | Standard deviation of the noise added for weak DP defense.                                             |
| `norm_bound` | Norm bound for the norm difference clipping defense.                                                   |
| `attack_method` | Attacking schemes used for attacker and either be `blackbox` or `PGD`.                                 |
| `attack_case` | Wether or not to conduct edge-case attack, can be `edge-case`, `normal-case` or `almost-edge-case`.    |
| `model_replacement` | Used when `attack_method=PGD` to control if the attack is PGD with replacement or without replacement. |
| `project_frequency` | How frequent (in how many iterations) to project to the l2 norm ball in PGD attack.                    |
| `eps` | Radius the l2 norm ball in PGD attack.                                                                 |
| `adv_lr` | Learning rate of the attacker when conducting PGD attack.                                              |
| `poison_type` | Specify the backdoor for each dataset using `southwest` for CIFAR-10 and `ardis` for EMNIST.           |
| `device` | Specify the hardware to run the experiment.                                                            |


#### Sample command
Blackbox attack on Southwest Airline exmaple over CIFAR-10 dataset with vgg9 model where there is soft_hard filtering defense. The attacker participate in the fixed-frequency manner.
```
python main_averaging.py --fraction 0.1 \
--lr 0.02 \
--gamma 0.998 \
--num_nets 200 \
--fl_round 500 \
--part_nets_per_round 10 \
--local_train_period 2 \
--adversarial_local_training_period 2 \
--dataset cifar10 \
--model vgg9 \
--fl_mode fixed-freq \
--attacker_pool_size 100 \
--defense_method soft_hard \
--attack_method blackbox \
--attack_case edge-case \
--model_replacement False \
--project_frequency 10 \
--stddev 0.025 \
--eps 2 \
--adv_lr 0.02 \
--prox_attack False \
--poison_type southwest \
--norm_bound 2 \
--device=cuda:0
```
