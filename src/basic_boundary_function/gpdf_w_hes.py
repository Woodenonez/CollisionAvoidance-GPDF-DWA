import jax
import jax.numpy as jp
from jax import vmap, jit

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'cpu')

@jit
def pair_dist(x1: jax.Array, x2: jax.Array) -> jax.Array:
    return jp.linalg.norm(x2 - x1)

@jit
def cdist(x, y):
    return vmap(lambda x1: vmap(lambda y1: pair_dist(x1, y1))(y))(x)


def pair_diff(x1, x2):
    return x1 - x2

@jit
def cdiff(x, y):
    return vmap(lambda x1: vmap(lambda y1: pair_diff(x1, y1))(y))(x)

@jit
def cov(x1, x2):
    d = cdist(x1, x2)
    return jp.exp(-d / 0.2) # covariance matrix

@jit
def cov_grad(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    return (-jp.exp(-d / 0.2) / 0.2 / d).reshape((x1.shape[0], x2.shape[0], 1)) * pair_diff_values

@jit
def cov_hessian(x1, x2):
    pair_diff_values = cdiff(x1, x2)
    d = jp.linalg.norm(pair_diff_values, axis=2)
    h = (pair_diff_values.reshape((x1.shape[0], x2.shape[0], 2, 1)) @ pair_diff_values.reshape((x1.shape[0], x2.shape[0], 1, 2)))
    hessian_constant_1 = (jp.exp(-d / 0.2)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_2 = (1 / (0.2 ** 2 * d ** 2) + 1 / (0.2 * d ** 3)).reshape((x1.shape[0], x2.shape[0], 1, 1))
    hessian_constant_3 = (1 / (0.2 * d)).reshape((x1.shape[0], x2.shape[0], 1, 1)) * jp.eye(2).reshape((1, 1, 2, 2))
    return hessian_constant_1 * (hessian_constant_2 * h + hessian_constant_3)


@jit
def reverting_function(x):
    return -0.2 * jp.log(x)

@jit
def reverting_function_derivative(x):
    return -0.2 / x

@jit
def reverting_function_second_derivative(x):
    return 0.2 / x ** 2

@jit
def infer_gpdf_dis(model, coords, query):
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)
    return mean-0.0

@jit
def infer_gpdf(model, coords, query):
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)

    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)
    return mean-0.0, grad

@jit
def infer_gpdf_hes(model, coords, query):
    k = cov(query, coords)
    mu = k @ model
    mean = reverting_function(mu)

    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)

    hessian = (jp.moveaxis(mu_grad, 0, 1)
               @ jp.moveaxis(mu_grad, 0, 2)
               * reverting_function_second_derivative(mu)[:, :, None])
    covariance_hessian = cov_hessian(query, coords)
    mu_hessian = ((jp.moveaxis(covariance_hessian, 1, -1) @ model)[..., 0]
                  * reverting_function_derivative(mu)[:, :, None])
    hessian += mu_hessian
    return mean-0.0, grad, hessian

@jit
def infer_gpdf_grad(model, coords, query):
    k = cov(query, coords)
    mu = k @ model
    covariance_grad = cov_grad(query, coords)
    mu_grad = jp.moveaxis(covariance_grad, -1, 0) @ model
    grad = reverting_function_derivative(mu) * mu_grad
    norms = jp.linalg.norm(grad, axis=0, keepdims=True)
    grad = jp.where(norms != 0, grad, grad / jp.min(jp.abs(grad), axis=0))
    grad /= jp.linalg.norm(grad, axis=0, keepdims=True)
    return grad

@jit
def train_gpdf(coords) -> jax.Array:
    coords = jp.array(coords)
    K = cov(coords, coords)
    y = jp.ones((len(coords), 1)) # observed distance (which is the mean of the GP)
    model = jp.linalg.solve(K, y)
    return model

