## Message Passing Attention Networks for Document Understanding 
Code for the paper [Message Passing Attention Networks for Document Understanding](https://arxiv.org/pdf/1908.06267.pdf).

### Requirements
Code is written in Python 3.6 and requires:
* PyTorch 1.1
* gensim 3.8
* scikit-learn 0.21

### Word embeddings
Download and unzip the pre-trained word2vec vectors from the following link: https://code.google.com/p/word2vec/

### Run the model
For the simple model, run:

```
python mpad/main.py --path-to-embeddings path
```

where path points to the word2vec binary file (i.e., `GoogleNews-vectors-negative300.bin` file). 


For the hierarchical models, run:

```
python hierarchical_mpad/main.py --path-to-embeddings path --graph-of-sentences type
```

where type can take the values 'clique', 'path' or 'sentence_att', and each value corresponds to one of the three hierarchical models described in the paper. 

### Cite
Please cite our paper if you use this code:
```
@inproceedings{nikolentzos2020message,
  title={Message Passing Attention Networks for Document Understanding},
  author={Nikolentzos, Giannis and Tixier, Antoine Jean-Pierre and Vazirgiannis, Michalis},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

-----------

Provided for academic use only
