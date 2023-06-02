# Nicholas， 2018

## Dirichlet process mixture

### Dirichlet distribution

- Normalizing constant

### Dirichlet process: distribution over distributions

$$
G\sim DP(\alpha, G_0),X_n|G\sim G
$$

- Note
- Example

	- Comparison

### Sampling from DP

- Stick breaking viewpoint
- Chinese restaurant process: posterior of DP

	- Clustering effect

### Dirichlet process mixture model

- Background
Given a data set, and are told that it was 
generated from a mixture of Gaussian distributions. But no one has any idea how many Gaussians produced the data.
- Finite mixture models

	- Assumption
A finite mixture model assumes that the data come from a mixture of a finite number of distributions
	- Illustration

		- Infinite mixture models

### Inference for DPMM

- Procedure (infer)

	- Take a random guess
	- Calculate mean for each cluster

		- Let #clusters grow or shrink

	- Compute cluster assignment for each sample

		- Update cluster by sampling from the distribution

- Procedure (generate)

	- Generate a new label assignment

		- For unique assignment

	- Illustration

- Comparison between EM and Gibbs

## Main model

### Model components

- DPMM

$$
\begin{align} &M \sim \text{Gamma}(\Psi_1,\Psi_2) \\
&G|M \sim\text{DPMM}(M,G_0),G_0 =N(0,\sigma_\tau^2) \\
&\tau_i |G \sim G
\end{align}
$$

	- Prior specification

$$
\Psi_1=2,\Psi_2=0.1
$$

- BART

$$
m \sim \text{BART}(\alpha,\beta,k,J)
$$

	- Prior specification

$$
\alpha=0.95,\beta=2,k=2,J=200
$$
	- Assumptions

$$
y = \hat \mu_{AFT}+\epsilon,\epsilon \sim N(0,\hat\sigma^2_{AFT})
$$

		- Centered transformation

$$
y_i^{tr}=y_i\exp\{-\hat \mu_{AFT}\}
$$
		- Prior of terminal nodes

$$
\begin{align} \mu_{j,l}\sim N(0,\frac{4 \hat \sigma^2_{AFT}}{Jk^2})
\end{align}
$$

			- Prior of BART

$$
m(A,x)\sim N(0,\frac{4 \hat \sigma^2_{AFT}}{k^2})
$$

- Nonparametric AFT model

$$
\begin{align} &\log T_i=m(A_i,x_i)+W_i \\
&\sigma^2 \sim \kappa \nu / \chi^2_\nu,\kappa=\sigma^2_\tau \\
&W_i|\tau_i,\sigma^2 \sim N(\tau_i,\sigma^2)
\end{align}
$$

	- Prior specification

$$
P(Var(W|G,\sigma)\leq \hat \sigma^2_W)=0.5 \rightarrow \kappa=\hat\sigma_W^2/4,\sigma^2_\tau=\kappa
$$

### Posterior inference

- Blocked Gibbs sampling

## AFT model

### Accelerated failure time (AFT) model

$$
\begin{align} \log T_i=x_i^T\beta + \sigma \epsilon_i,\epsilon_i \sim f \text{ (completely unspecified)}
\end{align}
$$

- Expression

	- Hazard function

$$
\lambda(t/x)=\exp(\beta'x)\lambda_0(\exp(\beta'x)t)
$$
	- Survival function

$$
S_i(t)=S_0(e^{x_i^T \beta}t)
$$

- Common distributions of AFT models

	- Weibull distribution and exponential AFT model

$$
\begin{align} S(t)=\exp(-\lambda t^\alpha)
\end{align}
$$

		- Weibull distribution

$$
f(t;\lambda,\alpha)=\alpha \lambda t^{\alpha-1} \exp(-\lambda t^\alpha)
$$
		- Hazard function

$$
h(t)=\lambda \alpha t^{\alpha-1}
$$

			- Cumulated hazard function

		- 自由主题

	- Log-normal AFT model
	- Log-logistic distribution

- Estimation of AFT models

	- MLE estimation

- Discussion

	- Advantage

		- 

	- Statistical issues

		- Illustration

### Nonparametric AFT model

$$
\log(T)=m(A,x)+W, A\in\{0,1\}
$$

## Individual treatment effect

### Ratio in expected survival time

$$
\xi(x)=\frac{E(T|A=1,x,m))}{E(T|A=0,x,m)}=\exp(\theta(x))
$$

### Heterogeneity of treatment effect

$$
\begin{align} &D_i=P(\theta(x_i)\geq \bar \theta |y,\delta), \bar \theta = \sum \theta(x_i)/n \\
&D_i^*=\max\{1-2D_i,2D_i-1\}
\end{align}
$$

## Bayesian additive regression tree (BART)

### Building a sum of trees

$$
y=\sum^m_{j=1}g(x;T_j,M_j)+\epsilon, \epsilon \sim N(0,\sigma^2)
$$

- Binary regression tree

### regularization of prior

- Prior of tree

	- Prior of tree at depth d nonterminal

$$
d_j=\frac{\alpha}{(1+d)^\beta}, \alpha \in (0,1), \beta \geq 0
$$
	- The distribution on the splitting variable assignments at each interior node is the uniform prior on available variables
	- The distribution on the splitting rule assignment in each interior node, conditional on the splitting variable is the uniform prior on the discrete set of available splitting values

- Prior of mean

$$
y \rightarrow [y_{\min}=-0.5, y_{\max}=0.5] \Rightarrow \mu_{ij}\sim N(0,\sigma^2_\mu), \sigma_\mu=\frac{1}{2k\sqrt{m}}
$$
- Prior of σ

$$
\sigma^2 \sim \frac{\nu \lambda}{\chi^2_{\nu}}
$$

	- Chipman (2010)

$$
(\nu,q)=(3,0.9),P(\sigma<\hat\sigma)=q
$$

- Number of trees

$$
m=200
$$

### Posterior sampling

- backfitting MCMC algorithm

$$
p((T_1,M_1),...,(T_m,M_m),\sigma|y)
$$

	- Partial residual

$$
R_j=y-\sum_{k\neq j}g(x;T_k,M_k)
$$
	- Draw σ from full conditional

$$
\sigma|T_1,...,T_m,M_1,...,M_m,y
$$
	- Gibbs sampling

$$
\begin{align} &T_j |R_j,\sigma \\
&M_j|T_j,R_j,\sigma
\end{align}
$$

- Particle Gibbs algorithms

