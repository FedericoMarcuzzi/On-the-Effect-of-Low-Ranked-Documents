On the Effect of Low-Ranked Documents: A New Sampling Function for Selective Gradient Boosting
===============================

This code is for the SAC 2023 full paper [On the Effect of Low-Ranked Documents: A New Sampling Function for Selective Gradient Boosting](https://doi.org/10.1145/3555776.3577597).

Abstract
---

Learning to Rank is the task of learning a ranking function from a set of query-documents pairs. Generally, documents within a query are thousands but not all documents are informative for the learning phase. Different strategies were designed to select the most informative documents from the training set. However, most of them focused on reducing the size of the training set to speed up the learning phase, sacrificing effectiveness. A first attempt in this direction was achieved by Selective Gradient Boosting a learning algorithm that makes use of customisable sampling strategy to train effective ranking models. In this work, we propose a new sampling strategy called High_Low_Sampl for selecting negative examples applicable to Selective Gradient Boosting, without compromising model effectiveness. The proposed sampling strategy allows Selective Gradient Boosting to compose a new training set by selecting from the original one three document classes: the positive examples, high-ranked negative examples and low-ranked negative examples. The resulting dataset aims at minimizing the mis-ranking risk, i.e., enhancing the discriminative power of the learned model and maintaining generalisation to unseen instances. We demonstrated through an extensive experimental analysis on publicly available datasets, that the proposed selection algorithm is able to make the most of the negative examples within the training set and leads to models capable of obtaining statistically significant improvements in terms of NDCG, compared to the state of the art.

Implementation
---

The code implements SelGB and the two sampling functions *Sel_Samp* and *Higl_Low_Samp*.

Citation
---

```
@inproceedings{DBLP:conf/sac/LuccheseM023,
  author       = {Claudio Lucchese and
                  Federico Marcuzzi and
                  Salvatore Orlando},
  editor       = {Jiman Hong and
                  Maart Lanperne and
                  Juw Won Park and
                  Tom{\'{a}}s Cern{\'{y}} and
                  Hossain Shahriar},
  title        = {On the Effect of Low-Ranked Documents: {A} New Sampling Function for
                  Selective Gradient Boosting},
  booktitle    = {Proceedings of the 38th {ACM/SIGAPP} Symposium on Applied Computing,
                  {SAC} 2023, Tallinn, Estonia, March 27-31, 2023},
  pages        = {646--652},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3555776.3577597},
  doi          = {10.1145/3555776.3577597},
  timestamp    = {Fri, 21 Jul 2023 22:25:37 +0200},
  biburl       = {https://dblp.org/rec/conf/sac/LuccheseM023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
