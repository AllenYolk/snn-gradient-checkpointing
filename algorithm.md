# Algorithm

**Input**: a model $f$ with $L$ layers, a list of block type $\{\xi_i\}_{i=1}^M$ that will be checkpointed, a list of rules $\{\rho_j\}_{j=1}^{K}$ indicating how a GC block can be split

1. **Layer-wise GC**. For each submodule of $f$ whose type is in $\{\xi_i\}_{i=1}^M$: convert it to a GC block.

2. **Spatial Split**. Profile each GC block's memory usage and identify the critical block with the largest memory cost. Split it into several GC blocks according to $\{\rho_j\}_{j=1}^{K}$. Repeat this process until the overall memory cost cannot further reduce.

3. **Temporal Split**. Profile each GC block's memory usage and identify the critical block with the largest memory cost. Split it temporally (i.e. increase the number of its temporal chunks), yielding some temporal GC blocks. Repeat this process until the overall memory cost cannot further redue.

4. **Greedily Disabling**. Profile each GC block's forward time cost and sort them in descending order. Greedily try to revert the GC blocks to conventional blocks following this order; if the overall memory cost increases, cancel this reversion. End this process until all blocks have be taken into account.
