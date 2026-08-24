# Iterating Fuel Mass During Trajectory Simulation

The v0 implementation of mass iteration in the new AEIC is based on a simple naive correction with a set maximum number of iteration. In this case, the mass residual is specified as the actual fuel burned over the predicted total fuel mass:

$$
\delta = \frac{\text{Actual Fuel Burn} - M_f}{M_f}
$$

This correction can be improved using an approximation from the Breguet range equation. Assuming the aerodynamic and engine properties are negligibly changed by the mass variation, the original `_fly_iteration` call used $\delta$ percent more fuel than was loaded and flew a range, $R$:

$$
R \propto -c^*\ln\bigg(1 - \frac{M_f}{M_0}(1+\delta)\bigg)
$$

We call the initial iteration's fuel-to-initial mass fraction $\lambda$ such that:

$$
R \propto -c^*\ln\bigg(1 - \lambda(1+\delta)\bigg)
$$

We want to fly the same mission (same range) with some additional amount of fuel relative to the initial prediction, $\Delta = \text{Additional Fuel}/M_f$. This would correspond to:

$$
R \propto -c^*\ln\bigg(1 - \frac{1 + \Delta}{1/\lambda + \Delta}\bigg)
$$

Again, we assume the aero and engine properties remain constant such that $c^*\equiv const$. Using this, we set the ranges equal and get:

$$
1-\lambda(1+\delta) = 1 - \frac{1 +\Delta}{1/\lambda + \Delta}
$$

$$
\frac{1 + \Delta}{1/\lambda + \Delta} = \lambda(1+\delta)
$$

$$
1 + \Delta = \bigg(\frac{1}{\lambda} + \Delta\bigg)\lambda(1 + \delta)
$$

$$
1 + \Delta = (1 + \lambda\Delta)(1 + \delta)
$$

$$
1 + \Delta = 1 + \lambda\Delta + \delta + \delta\lambda\Delta
$$

$$
\Delta(1 - \lambda - \delta\lambda) = \delta
$$

$$
\Delta = \frac{\delta}{1 - \lambda(1 + \delta)}
$$

In the naive approach, we simply set $\Delta = \delta$. The difference between these methods is shown in the figure below:

```{image} ../../_static/Naive-vs-rangecorr.png
:align: center
```

We see that for cases where additional fuel is needed to complete the mission ($\delta > 0$), we must add more fuel than the naive approach suggests. Conversely, for missions with too much initial fuel ($\delta<0$), we can remove more fuel than predicted by the naive method.

However, when testing these implementations in `notebooks/mass_iteration.ipynb`, the range-corrected method was found to be overshooting the correction, leading to increased iterations. To address this, a simple arithmetic average of the naive and range-corrected methods is preferred.
