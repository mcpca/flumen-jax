import equinox
import jax
from equinox.nn import MLP, LSTMCell
from jax import numpy as jnp
from jax import random as jrd
from jaxtyping import Array, Float, PRNGKeyArray, UInt

from .typing import Output, State, RNNInput, Input, TimeIncrement


class Flumen(equinox.Module):
    encoder: MLP
    decoder: MLP
    cell: LSTMCell

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        output_dim: int,
        feature_dim: int,
        encoder_hsz: int,
        decoder_hsz: int,
        key: PRNGKeyArray,
    ):
        enc_key, lstm_key, dec_key = jrd.split(key, 3)

        self.encoder = MLP(
            in_size=state_dim,
            out_size=feature_dim,
            width_size=encoder_hsz,
            depth=2,
            activation=jnp.tanh,
            key=enc_key,
        )

        self.cell = LSTMCell(
            input_size=control_dim + 1,
            hidden_size=feature_dim,
            key=lstm_key,
        )

        self.decoder = MLP(
            in_size=feature_dim,
            out_size=output_dim,
            width_size=decoder_hsz,
            depth=2,
            activation=jnp.tanh,
            key=dec_key,
        )

    def __call__(
        self,
        initial_state: State,
        rnn_input: RNNInput,
        tau: TimeIncrement,
        len: UInt[Array, "1"],
    ) -> Output:
        h = self.encoder(initial_state)
        c = jnp.zeros_like(h)

        (h_last, _), h_seq = jax.lax.scan(
            lambda state, input: (self.cell(input, state), state[0]),
            (h, c),
            rnn_input,
        )

        h0 = h_seq[len - 1]
        h1 = jax.lax.select(len >= h_seq.shape[0], h_last, h_seq[len])

        return self.decoder((1 - tau) * h0 + tau * h1)

    def eval_trajectory(
        self,
        initial_state: State,
        u: Input,
        tau: Float[Array, "n_time_pts 1"],  # noqa: F722
        skips: UInt[Array, "n_time_pts"],  # noqa: F821
    ) -> Float[Array, "n_time_pts output_dim"]:  # noqa: F722
        h = self.encoder(initial_state)
        c = jnp.zeros_like(h)

        rnn_input_integer_steps = jnp.concatenate(
            (u, jnp.ones((u.shape[0], 1))), axis=-1
        )

        (_, _), rnn_state_seq = jax.lax.scan(
            lambda state, input: (self.cell(input, state), state),
            (h, c),
            rnn_input_integer_steps,
        )

        rnn_input_tau = jnp.concatenate((u[skips], tau), axis=-1)
        h0 = rnn_state_seq[0][skips, :]
        c0 = rnn_state_seq[1][skips, :]

        h1, _ = jax.vmap(self.cell)(rnn_input_tau, (h0, c0))

        return jax.vmap(
            lambda h0, h1, tau: self.decoder((1 - tau) * h0 + tau * h1)
        )(h0, h1, tau)
