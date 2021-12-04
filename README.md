# GCN and FCN Code

This repository contains code for a project running GCN or FCN on the CORA dataset.

The purpose was to compare the two methods and even combine them to carry out node classification tasks on graphs. The current project looks at CORA paper citation network data and looks to classify papers by their subject category.

To run the model on CORA data, first download it here: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz. Put the `cora.cites` and `cora.content` files in `data/cora/` at the root. You will have to create these subfolders.

To run the out of the box GCN model on the CORA data, use this command in the terminal: `python run.py data model`.

To customize the model type, edit `model-params.json` within `config` and choose between `FCN` and `GCN`. You can also change model parameters from within that config file.

To customize the graph embedding types, edit `data-params.json` within `config` and toggle inclusion of graph ad hoc features or node2vec embeddings.
