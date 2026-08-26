# MuSiQue-Ans data directory

This directory holds local copies of the **MuSiQue-Ans** train and development
splits used by AdaptiveHop. The dataset is not versioned in this repository.

## Obtain the data

The canonical source is the [official MuSiQue repository](https://github.com/StonyBrookNLP/musique),
which provides its own download script and lists the MuSiQue-Ans data files.

The loader accepts either the project-local filenames below or the official
release names that are currently present in this directory:

```text
data/musique_ans/musique_ans_v1.0_train.jsonl
data/musique_ans/musique_ans_v1.0_dev.jsonl
```

We validated the following MuSiQue-Ans targets from the supplied files:

| Split | 2-hop | 3-hop | 4-hop | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 14,376 | 4,387 | 1,175 | 19,938 |
| Dev | 1,252 | 760 | 405 | 2,417 |


## Licensing and use

MuSiQue has its own license and usage terms. Review the original release before
downloading, sharing, or using the data. Those terms are separate from this
repository's MIT license, which covers AdaptiveHop source code only.
