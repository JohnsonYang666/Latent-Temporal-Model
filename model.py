import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import random, lax
from jax.nn import identity,sigmoid, tanh, leaky_relu
from jax.nn.initializers import glorot_normal, normal
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict
from typing import Any, Callable, Tuple, List, Dict

class Encoder(nn.Module):
    latent_dim: int = 512
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jax.Array, key: jax.Array, use_running_average: bool = False) -> Tuple[jax.Array, jax.Array, jax.Array]:
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.Conv(features=512, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        mu = nn.Dense(self.latent_dim)(x)
        log_var = nn.Dense(self.latent_dim)(x)

        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(key, shape=std.shape)
        z = mu + std * eps
        return z, mu, log_var

class Decoder(nn.Module):
    latent_dim: int = 512
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, z: jax.Array, use_running_average: bool = False) -> jax.Array:
        x = nn.Dense(4 * 4 * 512)(z)
        x = self.act_fn(x)
        x = x.reshape((z.shape[0], 4, 4, 512))

        x = nn.ConvTranspose(features=256, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=use_running_average)(x)
        x = self.act_fn(x)

        x = nn.ConvTranspose(features=1, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.tanh(x)
        return x

class VAE(nn.Module):
    latent_dim: int = 512
    act_fn: Callable = nn.relu

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, act_fn=self.act_fn)
        self.decoder = Decoder(latent_dim=self.latent_dim, act_fn=self.act_fn)

    def __call__(self, x: jax.Array, key: jax.Array, use_running_average: bool = False):
        z, mu, log_var = self.encoder(x, key, use_running_average=use_running_average)
        x_recon = self.decoder(z, use_running_average=use_running_average)
        return x_recon, mu, log_var

    def encode_only(self, x: jax.Array, key: jax.Array, use_running_average: bool = False):
        return self.encoder(x, key, use_running_average=use_running_average)

    def decode_only(self, z: jax.Array, use_running_average: bool = True):
        return self.decoder(z, use_running_average=use_running_average)
    
    def count_params(self, params: dict):
        def _count(pytree):
            if isinstance(pytree, (dict, FrozenDict)):
                return sum(_count(v) for v in pytree.values())
            else:
                return np.prod(pytree.shape)
        encoder_params = params.get("encoder", {})
        decoder_params = params.get("decoder", {})
        encoder_count = _count(encoder_params)
        decoder_count = _count(decoder_params)
        total_count = encoder_count + decoder_count
        return encoder_count, decoder_count, total_count

class StackedGRU:
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, seed=0,
                activation=tanh, gate_fn=sigmoid, dense_activation=leaky_relu):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.key = random.PRNGKey(seed)
        self.activation = activation     # e.g., tanh, relu, gelu
        self.gate_fn = gate_fn           # e.g., sigmoid
        self.dense_activation = dense_activation # e.g., sigmoid

        self.params = {
            "gru": self.initialize_stacked_gru_params(),
            "dense": {
                "w": glorot_normal()(random.PRNGKey(seed + 100), (output_dim, hidden_dim)),
                "b": jnp.zeros((output_dim,))
            }
        }

    def initialize_stacked_gru_params(self):
        params_list = []
        key_layers = random.split(self.key, self.num_layers)
        for i in range(self.num_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            key = key_layers[i]
            params = self.initialize_gru_params(key, input_dim, self.hidden_dim)
            params_list.append(params)
        return params_list

    def initialize_gru_params(self, key, input_dim, hidden_dim):
        keys = random.split(key, 9)
        weight_init = glorot_normal()
        bias_init = normal(stddev=1e-2)

        Wz = weight_init(keys[0], (hidden_dim, input_dim))
        Uz = weight_init(keys[1], (hidden_dim, hidden_dim))
        bz = bias_init(keys[2], (hidden_dim,))
        Wr = weight_init(keys[3], (hidden_dim, input_dim))
        Ur = weight_init(keys[4], (hidden_dim, hidden_dim))
        br = bias_init(keys[5], (hidden_dim,))
        Wh = weight_init(keys[6], (hidden_dim, input_dim))
        Uh = weight_init(keys[7], (hidden_dim, hidden_dim))
        bh = bias_init(keys[8], (hidden_dim,))
        return {
            'Wz': Wz, 'Uz': Uz, 'bz': bz,
            'Wr': Wr, 'Ur': Ur, 'br': br,
            'Wh': Wh, 'Uh': Uh, 'bh': bh
        }

    def gru_step(self, params, h_prev, x_t):
        z_t = self.gate_fn(jnp.dot(params['Wz'], x_t) + jnp.dot(params['Uz'], h_prev) + params['bz'])
        r_t = self.gate_fn(jnp.dot(params['Wr'], x_t) + jnp.dot(params['Ur'], h_prev) + params['br'])
        h_tilde_t = self.activation(jnp.dot(params['Wh'], x_t) + jnp.dot(params['Uh'], (r_t * h_prev)) + params['bh'])
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde_t
        return h_t, h_t

    def gru_sequence_with_params(self, params, inputs):
        initial_h = jnp.zeros((self.hidden_dim,))
        scan_fn = lambda carry, x: self.gru_step(params, carry, x)
        _, outputs_h = lax.scan(scan_fn, initial_h, inputs)
        return outputs_h  # shape: (seq_len, hidden_dim)
    
    def forward(self, batch_input, params=None):
        if params is None:
            params = self.params
        gru_params = params["gru"]
        dense_params = params["dense"]
        return jax.vmap(lambda x: self.forward_single_sample(x, gru_params, dense_params))(batch_input)
   
    def forward_single_sample(self, inputs, gru_params, dense_params):
        out = inputs # (seq_len, input_dim)
        for p in gru_params:
            out = self.gru_sequence_with_params(p, out)
        return self.timewise_dense(out, dense_params)  # (seq_len, output_dim)
    
    def timewise_dense(self, h_seq, dense_params):
        # Project hidden states at all time steps: (seq_len, hidden_dim) → (seq_len, output_dim)
        w, b = dense_params["w"], dense_params["b"]
        return jax.vmap(lambda h: self.dense_activation(jnp.dot(w, h) + b))(h_seq)
    
    def count_params(self):
        total = 0
        # Count GRU layer parameters
        for layer_params in self.params["gru"]:
            for p in layer_params.values():
                total += np.prod(p.shape)
        # Count dense layer parameters
        for p in self.params["dense"].values():
            total += np.prod(p.shape)
        return total

class StackedGRUFiLM(StackedGRU):
    """
    FiLM
        u_t = act( (1 + gamma(z)) * (h_t @ W_h) + beta(z) )
        y_t = u_t @ W_out + b_out
    where gamma(z) = z @ Wg + bg :tanh for limited amplitude
         beta(z)  = z @ Wb + bb
    """
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim,
                 num_layers=1, seed=0,
                 activation=tanh, gate_fn=sigmoid, dense_activation=leaky_relu,
                 gamma_activation=tanh,
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.key = random.PRNGKey(seed)
        self.activation = activation
        self.gate_fn = gate_fn
        self.dense_activation = dense_activation
        self.latent_dim = latent_dim
        self.gamma_activation = gamma_activation

        self.params = {
            "gru": self.initialize_stacked_gru_params(),
            "film": self.init_film_params(seed + 300),
            "out": {
                "w": glorot_normal()(random.PRNGKey(seed + 301), (hidden_dim, output_dim)),
                "b": jnp.zeros((output_dim,))
            }
        }

    def init_film_params(self, seed):
        k1, k2, k3 = random.split(random.PRNGKey(seed), 3)
        film_params = {
            "W_h": glorot_normal()(k1, (self.hidden_dim, self.hidden_dim)),  # (H, H)
            "Wg": glorot_normal()(k2, (self.latent_dim, self.hidden_dim)),   # (L, H)
            "bg": jnp.zeros((self.hidden_dim,)),
            "Wb": glorot_normal()(k3, (self.latent_dim, self.hidden_dim)),   # (L, H)
            "bb": jnp.zeros((self.hidden_dim,))
        }
        return film_params

    def film_block(self, h: jnp.ndarray, z: jnp.ndarray, params: dict) -> jnp.ndarray:
        """
        h: (H,)  z: (L,)
        return: (H,)
        """
        W_h, Wg, bg, Wb, bb = params["W_h"], params["Wg"], params["bg"], params["Wb"], params["bb"]
        h_proj = jnp.dot(h, W_h)                        # (H,)
        gamma = jnp.dot(z, Wg) + bg                     # (H,)
        if self.gamma_activation is not None:
            gamma = self.gamma_activation(gamma)
        beta  = jnp.dot(z, Wb) + bb                     # (H,)
        preact = (1.0 + gamma) * h_proj + beta          # (H,)
        return self.dense_activation(preact)            # (H,)

    def forward(self, batch_input: jnp.ndarray, batch_z: jnp.ndarray, params: dict = None):
        """
        batch_input: (B, T, input_dim)
        batch_z:     (B, latent_dim)
        return:      (B, T, output_dim)
        """
        if params is None:
            params = self.params
        gru_params = params["gru"]
        film_params = params["film"]
        out_params  = params["out"]

        return jax.vmap(self.forward_single_sample, in_axes=(0, 0, None, None, None))(
            batch_input, batch_z, gru_params, film_params, out_params
        )

    def forward_single_sample(self, inputs: jnp.ndarray, z: jnp.ndarray,
                              gru_params: list, film_params: dict, out_params: dict):
        out = inputs  # shape: (seq_len, input_dim)
        for p in gru_params:
            out = self.gru_sequence_with_params(p, out)
        def step(h_t):
            u_t = self.film_block(h_t, z, film_params)     # (H,)
            y_t = jnp.dot(u_t, out_params["w"]) + out_params["b"]  # (out_dim,)
            return y_t

        return jax.vmap(step)(out)  # (T, output_dim)

    def count_params(self):
        total = 0
        for layer_params in self.params["gru"]:
            for p in layer_params.values():
                total += int(np.prod(p.shape))
        for p in self.params["film"].values():
            total += int(np.prod(p.shape))
        for p in self.params["out"].values():
            total += int(np.prod(p.shape))
        return total

class JointModel:
    def __init__(self, vae_model: nn.Module, gru_model: StackedGRU, learning_rate: float = 1e-3):
        self.vae = vae_model
        self.gru = gru_model
        self.lr = learning_rate

        # 单独为 VAE 初始化 optax 优化器
        self.vae_optimizer = optax.adam(self.lr)
        self.gru_optimizer = optax.adam(self.lr)

    def init(self, rng: jax.Array, image_shape=(128, 128, 1)):
        rng, vae_rng, gru_rng = jax.random.split(rng, 3)

        dummy_image = jnp.ones((1,) + image_shape)
        vae_variables = self.vae.init(vae_rng, dummy_image, vae_rng)
        vae_params = vae_variables["params"]
        vae_batch_stats = vae_variables.get("batch_stats", {})
        gru_params = self.gru.params

        vae_opt_state = self.vae_optimizer.init(vae_params)
        gru_opt_state = self.gru_optimizer.init(gru_params)

        params = {"vae": vae_params, "gru": gru_params}
        opt_states = {"vae": vae_opt_state, "gru": gru_opt_state}
        batch_stats = {"vae": vae_batch_stats}

        return params, opt_states, batch_stats, rng

    def count_params(self, params_dict: Dict[str, Any]):
        gru_param_count = self.gru.count_params()
        encoder_count, decoder_count, vae_param_count = self.vae.count_params(params_dict['vae'])
        total = gru_param_count + vae_param_count
        print(f"[GRU] Total parameters: {gru_param_count:,}")
        print(f"[VAE] Encoder parameters: {encoder_count:,}")
        print(f"[VAE] Decoder parameters: {decoder_count:,}")
        print(f"[VAE] Total parameters:   {vae_param_count:,}")
        print(f"[Total] Total parameters: {total:,}")
        return total

def train_joint_model_mu(model: JointModel,
    train_set: Dict[str, jnp.ndarray],
    params_dict: Dict[str, Any],
    opt_states_dict: Dict[str, Any],
    batch_stats_dict: Dict[str,Any],
    rng: jax.Array,
    num_epochs: int = 100,
    batch_size: int = 64,
    patience: int = 10,
    min_delta: float = 1e-4):

    num_samples = train_set['input'].shape[0]
    optimizer = optax.adam(model.lr)

    def compute_loss(model, vae_params, gru_params, vae_batch_stats, batch_images, batch_input, batch_target, rng):
        vae_vars = {"params": vae_params, "batch_stats": vae_batch_stats}

        (z_latent, mu, log_var), updated_state = model.vae.apply(
            vae_vars,
            batch_images,
            rng,
            method=VAE.encode_only,
            mutable=["batch_stats"],
            use_running_average=False
        )
        new_batch_stats = updated_state["batch_stats"]

        preds = model.gru.forward(batch_input, mu, gru_params)

        recon_images = model.vae.apply(
            {"params": vae_params, "batch_stats": vae_batch_stats},
            z_latent,
            method=VAE.decode_only,
            mutable=False,
            use_running_average=True
        )

        recon_loss = jnp.mean(jnp.sum((batch_images - recon_images) ** 2, axis=(1, 2, 3)))
        kl_div = -0.5 * jnp.sum(1 + log_var - mu ** 2 - jnp.exp(log_var), axis=1)
        kl_loss = jnp.mean(kl_div)
        pde_loss = jnp.mean((preds - batch_target) ** 2)

        total_loss = recon_loss + kl_loss + pde_loss
        return total_loss, (recon_loss, kl_loss, pde_loss, new_batch_stats)


    @jax.jit
    def update_step(vae_params, gru_params, vae_batch_stats, vae_opt_state, gru_opt_state, batch_images, batch_input, batch_target, rng):
        grad_fn = jax.value_and_grad(compute_loss, argnums=(1, 2), has_aux=True)
        (loss, (recon_loss, kl_loss, pde_loss, new_batch_stats)), (vae_grads, gru_grads) = grad_fn(
        model, vae_params, gru_params, vae_batch_stats, batch_images, batch_input, batch_target, rng
        )

        vae_updates, vae_opt_state = optimizer.update(vae_grads, vae_opt_state)
        gru_updates, gru_opt_state = optimizer.update(gru_grads, gru_opt_state)

        vae_params = optax.apply_updates(vae_params, vae_updates)
        gru_params = optax.apply_updates(gru_params, gru_updates)

        return vae_params, gru_params, new_batch_stats, vae_opt_state, gru_opt_state, loss, recon_loss, kl_loss, pde_loss


    def should_early_stop(loss_curve, patience, min_delta):
        if len(loss_curve) < patience + 1:
            return False
        recent = loss_curve[-(patience + 1):]
        deltas = [abs(recent[i + 1] - recent[i]) for i in range(patience)]
        return all(delta < min_delta for delta in deltas)

    vae_params, gru_params = params_dict['vae'], params_dict['gru']
    vae_batch_stats = batch_stats_dict['vae']
    vae_opt_state, gru_opt_state = opt_states_dict['vae'], opt_states_dict['gru']

    loss_history = {"total": [], "recon": [], "kl": [], "pde": []}
    for epoch in range(num_epochs):
        perm = jax.random.permutation(rng, num_samples)
        epoch_total_loss, epoch_recon_loss, epoch_kl_loss, epoch_pde_loss = 0.0, 0.0, 0.0, 0.0
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_images = train_set['image'][idx]
            batch_input = train_set['input'][idx]
            batch_target = train_set['output'][idx]

            rng, step_rng = jax.random.split(rng)
            vae_params, gru_params, vae_batch_stats, vae_opt_state, gru_opt_state, loss, recon_loss, kl_loss, pde_loss = update_step(
                vae_params, gru_params, vae_batch_stats, vae_opt_state, gru_opt_state,
                batch_images, batch_input, batch_target, step_rng)
            epoch_total_loss += float(loss)
            epoch_recon_loss += float(recon_loss)
            epoch_kl_loss += float(kl_loss)
            epoch_pde_loss += float(pde_loss)

        steps_per_epoch = num_samples // batch_size
        avg_total = epoch_total_loss / steps_per_epoch
        avg_recon = epoch_recon_loss / steps_per_epoch
        avg_kl = epoch_kl_loss / steps_per_epoch
        avg_pde = epoch_pde_loss / steps_per_epoch

        print(f"Epoch {epoch + 1} | Total: {avg_total:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | PDE: {avg_pde:.4f}")

        loss_history['total'].append(avg_total)
        loss_history['recon'].append(avg_recon)
        loss_history['kl'].append(avg_kl)
        loss_history['pde'].append(avg_pde)

        if should_early_stop(loss_history, patience, min_delta):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    final_params = {'vae': vae_params, 'gru': gru_params}
    final_opt_states = {'vae': vae_opt_state, 'gru': gru_opt_state}
    final_batch_stats = {'vae': vae_batch_stats}
    return final_params, final_opt_states, final_batch_stats, loss_history