from jaxtyping import Array, Float, UInt

State = Float[Array, "state_dim"]
Output = Float[Array, "output_dim"]
Input = Float[Array, "seq_len control_dim"]
RNNInput = Float[Array, "seq_len control_dim+1"]
TimeIncrement = Float[Array, "1"]

BatchedOutput = Float[Array, "batch output_dim"]
BatchedState = Float[Array, "batch state_dim"]
BatchedRNNInput = Float[Array, "batch seq_len control_dim+1"]
BatchedTimeIncrement = Float[Array, "batch 1"]
BatchLengths = UInt[Array, "batch 1"]

Inputs = tuple[
    BatchedState, BatchedRNNInput, BatchedTimeIncrement, BatchLengths
]
