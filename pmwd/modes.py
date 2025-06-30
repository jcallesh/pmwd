from functools import partial

from jax import jit, checkpoint, custom_vjp
from jax import random
import jax.numpy as jnp

from pmwd.boltzmann import linear_power, linear_transfer
from pmwd.pm_util import fftfreq, fftfwd, fftinv


#TODO follow pmesh to fill the modes in Fourier space
@partial(jit, static_argnames=('real', 'unit_abs'))
def white_noise(seed, conf, real=False, unit_abs=False):
    """White noise Fourier or real modes.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random number generator.
    conf : Configuration
    real : bool, optional
        Whether to return real or Fourier modes.
    unit_abs : bool, optional
        Whether to set the absolute values to 1.

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        White noise Fourier or real modes, both dimensionless with zero mean and unit
        variance.

    """
    key = random.PRNGKey(seed)

    # sample linear modes on Lagrangian particle grid
    modes = random.normal(key, shape=conf.ptcl_grid_shape, dtype=conf.float_dtype)

    if real and not unit_abs:
        return modes

    modes = fftfwd(modes, norm='ortho')

    if unit_abs:
        modes /= jnp.abs(modes)

    if real:
        modes = fftinv(modes, shape=conf.ptcl_grid_shape, norm='ortho')

    return modes


@custom_vjp
def _safe_sqrt(x):
    return jnp.sqrt(x)

def _safe_sqrt_fwd(x):
    y = _safe_sqrt(x)
    return y, y

def _safe_sqrt_bwd(y, y_cot):
    x_cot = jnp.where(y != 0, 0.5 / y * y_cot, 0)
    return (x_cot,)

_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)

# JC: Utilities for padding and unpadding the modes to avoid cluttering the equilateral case
def _safe_pad(field, conf):
    # TF: padding for antialiasing (factor of (3/2)**3. for the change in dimension)
    field = jnp.fft.fftshift(field,axes=[0,1])
    field = jnp.pad(field, ((conf.ptcl_grid_shape[0]//4,conf.ptcl_grid_shape[0]//4),(conf.ptcl_grid_shape[1]//4,conf.ptcl_grid_shape[1]//4),(0,conf.ptcl_grid_shape[2]//4))) * (3/2)**3.
    field = jnp.fft.ifftshift(field,axes=[0,1])
    return field

def _safe_unpad(field, conf):   
    # TF: downsampling (factor of (3/2)**3. for the change in dimension)
    field = jnp.fft.fftshift(field,axes=[0,1])
    field = field[conf.ptcl_grid_shape[0]//4:-conf.ptcl_grid_shape[0]//4, conf.ptcl_grid_shape[1]//4:-conf.ptcl_grid_shape[1]//4,:-conf.ptcl_grid_shape[2]//4] / (3/2)**3.
    field = jnp.fft.ifftshift(field,axes=[0,1])
    return field


@partial(jit, static_argnums=4)
@partial(checkpoint, static_argnums=4)
def _linear_modes(modes, cosmo, conf, a=None, real=False):
    kvec = fftfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)
    k = jnp.sqrt(sum(k**2 for k in kvec))

    if a is not None:
        a = jnp.asarray(a, dtype=conf.float_dtype)

    if jnp.isrealobj(modes):
        modes = fftfwd(modes, norm='ortho')
    
    if (cosmo.f_nl_loc_ is not None) and (cosmo.f_nl_equi_ is None):
        print("Computing local non-Gaussian ICs...")
        Tlin = linear_transfer(k, a, cosmo, conf)*k*k
        Pprim = 2*jnp.pi**2. * cosmo.A_s * (k/cosmo.k_pivot)**(cosmo.n_s-1.)\
                    * k**(-3.)
        Pprim = Pprim.at[0,0,0].set(0.)
        
        modes *= _safe_sqrt(Pprim / conf.ptcl_cell_vol)
        
        modes = fftinv(modes, norm='ortho')
        modes = jnp.fft.rfftn(modes)

        # TF: padding for antialiasing (factor of (3/2)**3. for the change in dimension)
        modes_NG = jnp.fft.fftshift(modes,axes=[0,1])
        modes_NG = jnp.pad(modes_NG, ((conf.ptcl_grid_shape[0]//4,conf.ptcl_grid_shape[0]//4),(conf.ptcl_grid_shape[1]//4,conf.ptcl_grid_shape[1]//4),(0,conf.ptcl_grid_shape[2]//4))) * (3/2)**3.
        modes_NG = jnp.fft.ifftshift(modes_NG,axes=[0,1])
        
        # TF: square the modes in real space
        modes_NG = jnp.fft.rfftn(jnp.fft.irfftn(modes_NG)**2.) 
        
        # TF: downsampling (factor of (3/2)**3. for the change in dimension)
        modes_NG = jnp.fft.fftshift(modes_NG,axes=[0,1])
        modes_NG = modes_NG[conf.ptcl_grid_shape[0]//4:-conf.ptcl_grid_shape[0]//4, conf.ptcl_grid_shape[1]//4:-conf.ptcl_grid_shape[1]//4,:-conf.ptcl_grid_shape[2]//4] / (3/2)**3.
        modes_NG = jnp.fft.ifftshift(modes_NG,axes=[0,1])
        
        # TF: add to the gaussian modes, factor of 3/5 is because we are generating \zeta and f_nl is defined for \Phi
        modes = jnp.fft.irfftn(modes)
        modes_NG = jnp.fft.irfftn(modes_NG)
        modes = modes + 3/5 * cosmo.f_nl_loc * (modes_NG - jnp.mean(modes_NG))
        modes = modes.astype(conf.float_dtype)

        # TF: apply transfer function
        modes = fftfwd(modes, norm='ortho')
        modes *= Tlin * conf.box_vol / jnp.sqrt(conf.ptcl_num)

    elif (cosmo.f_nl_equi_ is not None) and (cosmo.f_nl_loc_ is None):
        print("Computing equilateral non-Gaussian ICs...")
        # JC: Adapted from TF's implementation to extend PNG to the equilateral non-Gaussian case using Scoccimarro's separable method. This approach follows the 2LPTPNG code utilized by Quijote_PNG.
        Tlin = linear_transfer(k, a, cosmo, conf)*k*k
        Pprim = 2*jnp.pi**2. * cosmo.A_s * (k/cosmo.k_pivot)**(cosmo.n_s-1.)\
                    * k**(-3.)
        Pprim = Pprim.at[0,0,0].set(0.)

        modes *=  _safe_sqrt(Pprim / conf.ptcl_cell_vol)

        modes = fftinv(modes, norm='ortho')
        modes = jnp.fft.rfftn(modes)

        # JC: Set k's exponent for ns!=1
        kmag2 = sum(kk**2 for kk in kvec)
        k_2_over_3 = jnp.power(kmag2, (4. - cosmo.n_s)/3.)
        k_1_over_3 = jnp.power(kmag2, (4. - cosmo.n_s)/6.)

        # JC: lets start with the symetric kernel K_{12}^{2nd} --> FFT( iFFT( k^{1/3}phi(k) )^2 )/ k^{2/3}
        modes_sym = _safe_pad(k_1_over_3*modes, conf)
        modes_sym = jnp.fft.rfftn(jnp.fft.irfftn(modes_sym)**2.) 
        modes_sym = _safe_unpad(modes_sym, conf)
        modes_sym = modes_sym/k_2_over_3
        modes_sym = modes_sym.at[0, 0, 0].set(0. + 0.j)

        # JC: now the third genertor scalar kernel K_{12}^I -->  FFT(  iFFT(phi)*iFFT(k^{1/3}phi(k)) )/ k^{1/3}
        modes_sca = _safe_pad(k_1_over_3*modes, conf)
        modes_phi = _safe_pad(modes, conf) #we will re-use for the next kernels
        modes_sca = jnp.fft.rfftn(jnp.fft.irfftn(modes_sca)*jnp.fft.irfftn(modes_phi)) 
        modes_sca = _safe_unpad(modes_sca, conf)
        modes_sca = modes_sca/k_1_over_3
        modes_sca = modes_sca.at[0, 0, 0].set(0. + 0.j)

        # JC: continue with the nabla kernel K_{12}^{II} -->  FFT(  iFFT(phi)*iFFT(k^{2/3}phi(k)) )/ k^{2/3}
        modes_nab = _safe_pad(k_2_over_3*modes, conf)
        modes_nab = jnp.fft.rfftn(jnp.fft.irfftn(modes_nab)*jnp.fft.irfftn(modes_phi)) 
        modes_nab = _safe_unpad(modes_nab, conf)
        modes_nab = modes_nab/k_2_over_3
        modes_nab = modes_nab.at[0, 0, 0].set(0. + 0.j)

        # JC: finally K_{12} is the usual local term
        modes_pot = jnp.fft.rfftn(jnp.fft.irfftn(modes_phi)**2.) 
        modes_pot = _safe_unpad(modes_pot, conf)
        modes_pot = modes_pot.at[0, 0, 0].set(0. + 0.j)

        # JC: add all kernels to the gaussian modes and correct by the 3/5 factor
        modes = modes + 3./5.*cosmo.f_nl_equi_*( -3.* modes_pot - 2.* modes_sym + 4.* modes_sca + 2.* modes_nab )
        modes = jnp.fft.irfftn(modes)
        modes = modes.astype(conf.float_dtype)

        # JC: apply transfer function
        modes = fftfwd(modes, norm='ortho')
        modes *= Tlin * conf.box_vol / jnp.sqrt(conf.ptcl_num)

    else:
        print("Computing Gaussian ICs...")
        Plin = linear_power(k, a, cosmo, conf)
        modes *= _safe_sqrt(Plin * conf.box_vol)

    if real:
        modes = fftinv(modes, shape=conf.ptcl_grid_shape, norm=conf.ptcl_spacing)

    return modes

def linear_modes(modes, cosmo, conf, a=None, real=False):
    """Linear matter overdensity Fourier or real modes.

    Parameters
    ----------
    modes : jax.Array
        Fourier or real modes with white noise prior.
    cosmo : Cosmology
    conf : Configuration
    a : float or None, optional
        Scale factors. Default (None) is to not scale the output modes by growth.
    real : bool, optional
        Whether to return real or Fourier modes.

    Returns
    -------
    modes : jax.Array of conf.float_dtype
        Linear matter overdensity Fourier or real modes, in [L^3] or dimensionless,
        respectively.

    Notes
    -----

    .. math::

        \delta(\mathbf{k}) = \sqrt{V P_\mathrm{lin}(k)} \omega(\mathbf{k})

    """
    return _linear_modes(modes, cosmo, conf, a, real)

# JC: # *right now this function has a problem with white_noise_fixed and the @jit decorator. the error is --> jax.errors.NonConcreteBooleanIndexError(tracer).
# This is because the slicing kxx_m[primary] in modes.at[...].set[...] is not supported for non boolean slicing.
def _hermitian_symmetry(modes, conf):
    '''
        This function enforces the condition:  modes(-k) = modes*(k)
        where -k = (-kx, -ky, -kz).

    '''

    grid = conf.ptcl_grid_shape
    middle = grid[2] // 2

    x = jnp.arange(grid[0])
    y = jnp.arange(grid[1])
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # we only care when kz==0 or kz==middle
    for kzz in [0, middle]:

        kx = jnp.where(X > middle, X - grid[0], X)
        ky = jnp.where(Y > middle, Y - grid[1], Y)
        
        kxx_m = jnp.where(kx > 0, grid[0] - kx, -kx)
        kyy_m = jnp.where(ky > 0, grid[1] - ky, -ky)
        
        diag = (X == kxx_m) & (Y == kyy_m)

        primary = ((X < kxx_m) | ((X == kxx_m) & (Y < kyy_m))) & (~diag)
        
        modes = modes.at[kxx_m[primary], kyy_m[primary], kzz].set(
            jnp.conjugate(modes[X[primary], Y[primary], kzz])
        )

        modes = modes.at[X[diag], Y[diag], kzz].set(
            jnp.real(modes[X[diag], Y[diag], kzz]) + 0j
        )

    modes = modes.at[0, 0, 0].set(0.0 + 0.0j)
    return modes

# JC: to set fixed initial conditions:
#@partial(jit, static_argnames=('real', 'unit_abs'))
def white_noise_fixed(seed, conf, real=False, unit_abs=False):
    """White noise Fourier or real modes, with random phases and fixed amplitude.

    Parameters
    ----------
    seed : int
        Seed for the pseudo-random number generator.
    conf : Configuration
    real : bool, optional
        Whether to return real or Fourier modes.
    unit_abs : bool, optional
        Whether to set the absolute values to 1.

    Returns
    -------
    modes : jax.Array
    """
    key = random.PRNGKey(seed)
    nx, ny, nz = conf.ptcl_grid_shape

    phases = random.uniform(key, shape=(nx, ny, nz//2+1), dtype=conf.float_dtype) * 2 * jnp.pi
    modes = jnp.cos(phases) + 1.j*jnp.sin(phases) #jnp.exp(1.j * phases)
    
    modes = _hermitian_symmetry(modes, conf)

    if unit_abs:
        modes /= jnp.abs(modes)

    if real:
        modes = fftinv(modes, shape=conf.ptcl_grid_shape, norm='ortho')

    return modes
