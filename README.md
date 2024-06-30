# Metric Learning for Clifford Group Equivariant Neural Networks

**Authors:** Riccardo Ali, Paulina Kulyte

## Abstract
This study introduces a method to enhance Clifford Group Equivariant Neural Networks (CGENNs) by incorporating metric learning, particularly focusing on transitioning from static, diagonal metrics to dynamic, learnable metrics through the eigenvalue decomposition. CGENNs have been recognized for their capability to process complex data representations effectively, leveraging the mathematical framework of Clifford Algebras. However, these networks often face challenges in higher-dimensional spaces where the metrics are undefined a priori. Addressing this, our method not only optimizes a distance function directly from the dataset but also extends the network's capabilities to employ non-diagonal metrics through eigenvalue decomposition. The method allows for the accurate modeling of data relationships, acknowledging the intricate correlations in real-world data that diagonal metrics typically oversimplify. The results obtained show improved performance in simulating complex physical systems and analyzing high-dimensional datasets, advancing the creation of neural networks adept at comprehending complex data relationships.


## Code

The repo builds upon: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks. The datasets for the experiments can be obtained following the guidelines in the same link.

Implementation of metric learning:
- `algebra/`: Contains the Clifford algebra and Metric implementation.
- `engineer/`: Contains the training and evaluation scripts, including metric activation.
- `models/`: Contains model and layer implementations, which takes metric as argument.
- `engineer/trainer.py` line 387: Controls when to activate metric learning as a percentage of the total learning steps.

## Experiments

### O(3) Experiment: Signed Volumes
```python o3.py -C configs/engineer/trainer.yaml -C configs/optimizer/adam.yaml -C configs/dataset/o3.yaml -C configs/model/o3_cgmlp.yaml --trainer.max_steps=131072 --trainer.val_check_interval=1024 --dataset.batch_size=128 --dataset.num_samples=65536 --model.hidden_features=96 --model.num_layers=4 --optimizer.lr=0.001 --save_name=<NAME>```

### O(3) Experiment: $n$-body.
```python nbody.py -C configs/engineer/trainer.yaml -C configs/optimizer/adam.yaml -C configs/dataset/nbody.yaml -C configs/model/nbody_cggnn.yaml --trainer.val_check_interval=128 --trainer.max_steps=131072 --dataset.batch_size=100 --dataset.num_samples=3000 --optimizer.lr=0.004 --optimizer.weight_decay=0.0001 --save_name=<NAME>```

### O(1, 3) Experiment: Top Tagging
```CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=1  top_tagging.py -C configs/engineer/trainer.yaml -C configs/optimizer/adam.yaml -C configs/dataset/top_tagging.yaml -C configs/model/lorentz_cggnn.yaml --trainer.max_steps=331126 --trainer.val_check_interval=8192 --dataset.batch_size=32 --optimizer.lr=0.001 --save_name=<NAME>```


## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{
    ali2024metric,
    title={Metric Learning for Clifford Group Equivariant Neural Networks},
    author={Riccardo Ali and Paulina Kulyt{\.{e}} and Haitz S{\'a}ez de Oc{\'a}riz Borde and Pietro Lio},
    booktitle={ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling},
    year={2024},
    url={https://openreview.net/forum?id=4tw1U41t6Q}
}
```