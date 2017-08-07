[![Build Status](https://travis-ci.org/gokceneraslan/tensorsne.svg?branch=master)](https://travis-ci.org/gokceneraslan/tensorsne)
### A Fast Barnes-Hut tSNE implementation with TensorFlow support

Barnes-Hut tSNE with following cool features:

- Fast nearest neighbor calculation  with parallelization and sparse matrix support through `nmslib`
- KL divergence and Barnes-Hut gradient approximation as a Tensorflow op
- Parametric and non-parametric Barnes-Hut tSNE in Tensorflow and Keras
