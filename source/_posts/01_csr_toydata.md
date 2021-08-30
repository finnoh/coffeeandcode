---
title: Create some Toy Data
author: Finn
date: '2021-08-30'
slug: 01csr
categories:
  - Python
  - code_snippet_repository
tags:
  - data
  - code_snippet_repository
  - simulation
  - basic
  - pandas
  - sklearn
  - numpy
code_lang: "Python"
description: "Code Snippet Repository: How to get yourself some Toy Data"
weight: 100
featured: yes
toc: true
widgets:
  - type: toc
    position: left
    # Whether to show the index of each heading
    index: true
    # Whether to collapse sub-headings when they are out-of-view
    collapsed: true
    # Maximum level of headings to show (1-6)
    depth: 4

  - type: recent_posts
    position: left
---

> I wrote a little Code Snippet that returns the "California Housing" dataset provided by [`sklearn`](https://scikit-learn.org/stable/index.html) as a named [`pandas`](https://pandas.pydata.org/docs/) dataframe. Ready for you to use in your applications.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

dict_data = fetch_california_housing()

list_names = dict_data.feature_names
list_names.extend(dict_data.target_names)

data = np.concatenate([dict_data.data, dict_data.target.reshape(-1,1)], axis=1)
df = pd.DataFrame(data=data, columns=list_names)
```

## Get yourself some Toy Data
Often times, we come up with an idea for a visualization or simply want to try out a new model we have heard off. In short: We need data. 

However, this can sometimes become a tedious problem as a real world dataset might need additional cleaning, which can be too much effort for a short test or playing around. Here, a toy dataset can come in handy. There are tons of toy-data sets out there, many good ones presented by the [`sklearn.dataset`](https://scikit-learn.org/stable/datasets/toy_dataset.html) module. In the example above I remodel the arrays provided by `fetch_california` into a `pd.DataFrame`.

For more information on the "California Housing" dataset visit [`sklearn.datasets.fetch_california`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html?highlight=california%20housing#sklearn.datasets.fetch_california_housing).
