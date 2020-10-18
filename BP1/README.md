## Back Propagation (BP)

Back Propagation (BP) is a widely used algorithm to adjust weights during the training process of neural networks. It first generates random weights and calculates the output layer, and then alters the weights with the gradient of the loss to weights calculated by the chain rule. Our implementation, based on Rodinia’s implementation, propagates backward from the last layer to avoid redundant computing in the chain rule. The original BP is from [“Understanding co-running behaviors on integrated CPU/GPU architectures,” TPDS, 2017]

