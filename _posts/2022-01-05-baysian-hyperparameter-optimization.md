---
title: A Step-by-Step Introduction to Bayesian Hyperparameter Optimization

author:
  name: Dominik Polzer
  link: https://github.com/polzerdo55862
date: 2022-01-05 18:32:00 -0500
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
---

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/title.png" style="width:100%">
</figure>

## Introduction <a name="introduction"/>

Hyperparameters are parameters that are set before the actual training to control the learning process. The decision tree requires a limit for the maximum number of nodes
of the tree; the polynomial regression the polynomial degree of the trained model; the support vector regression the kernel,
the regularization parameter $$C$$ and the margin of tolerance $$\epsilon$$. All these parameters influence the training process and thus the performance of the resulting model.

The search for optimal hyperparameters is called hyperparameter optimization, i.e. the search for the hyperparameter combination for which the trained model shows the best performance for the given data set.
Popular methods are Grid Search, Random Search and Bayesian Optimization. This article explains the differences between these
approaches and focuses on Bayesian Optimization. The decisive factor for choosing the right optimization method is in most cases the computational effort required to evaluate the different hyperparameter settings.

If we want to know how the resulting model performs for a specific hyperparameter combination, we have no choice but to build the model using one subset of the dataset and evaluate it using a second subset.
The algorithm, the selected hyperparameter settings and the size of the dataset determine how computationally expensive the model building process is. For so-called "costly" optimization processes, it is
worth going beyond the principle of "simple" trial and error.

For illustrative purposes, consider a simple regression problem that we would like to solve using polynomial regression. In the first step we
choose settings for the hyperparameters that seem plausible to us (e.g. based on prior knowledge). Most regression problems in engineering can be solved sufficiently well
with polynomial regression models using polynomial degrees of less than 10. Prior knowledge like this can be used to narrow down the hyperparameter space in advance.

For the following evaluation, we set the hyperparameter space to polynomial degrees between 1 - 20. To ensure that we identify the optimal hyperparameter value in
the defined hyperparameter space, we could simply build a model for each value within this range and evaluate it. In regression, we usually compare the absolute or squared error
of the test values from the predicted values of the model. The individual evaluation steps are explained in more detail below.

With a for-loop, we could perform the following steps for each possible hyperparameter and save the results of each run in a list:

1. Define the polynomial regression algorithm with the specified hyperparameter setting
2. Build the model (using the train data set)
3. Evaluate the model (using the validation or test data set)

Afterwards, we simply choose the polynomial degree for our model that shows the best performance in the evaluation process.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/grid_search_polynomial_regression_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Hyperparameter search: simple for-loop - so-called “costly” evaluation</b>
  </figcaption>
</figure>

This approach is certainly valid for the small dataset in the image and relatively simple polynomial regression models, but reaches its
limits when the datasets and hyperparameter space are large and more computationally expensive algorithms are used.

In order to reduce the computing power required to find the optimal hyperparameter settings,
Bayesian optimization uses the Bayes' theorem. In simple terms, the Bayes' theorem is used to calculate the probability of an event, based on its
association with another event [Hel19].

So if we know the probability of observing the event $$A$$ and $$B$$ independently of each other
(the so called priori probability) and the probability of event $$B$$ occurring given that $$A$$ is true (the so-called conditional probability) we are able
to calculate the probability of event $$A$$ occurring given that $$B$$ is true (conditional probability) as follows:

$$P(A| B) = \frac{P(B |  A) \cdot P(A)}{P(B)}$$

A popular application of the Bayes' theorem is the disease detection. For rapid tests, one is interested in how high the actual probability is,
that a positive tested person actually has the disease. [Fah16]

In the context of hyperparameter optimization, we would like to predict the probability distribution of the loss value for any possible hyperparameter
combination in the defined hyperparameter space. With the help of some calculated "true" values of the loss function, we would like to model the function of
the loss function over the entire hyperparameter space - the resulting model function is the so-called **Surrogate Function**.
For our example, we could calculate the resulting loss for a polynomial
degree of 2, 8 and 16 and use regression analysis to train a function that approximates the loss over the entire hyperparameter space from 1 to 20.

The figure shows an example of a posteriori Gaussian process. The data set used for training consists of 10 known points of the true function.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/gaussian_process_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Example posteriori gaussian process - Image by the author</b>
  </figcaption>
</figure>

In Gaussian Process Regression, the resulting model provides not only an approximation of the true function (the mean function of the Gaussian process) but also a measurement of the
uncertainty (the covariance) of the model for each $$x$$. Simply put, the smaller the standard deviation (here: grey background) at a point in the function, the more certain the
model is that the mean value (here: black curve) represents a good approximation of the real loss value. If one would like to increase the accuracy of the approximation,
we could simply increase the size of the training data set. Since we can specifically choose some hyperparameter combinations and calculate the resulting loss of the model,
it's worth to think first what sampling point would probably result in the highest model improvement. A popular way to find the next sampling point, is to use the
uncertainty of the model as basis for the decision.

These and similar considerations are mapped into an **Acquisition Function**, which is the basis for choosing the next sampling point(s).

The following sections of the article will illustrate this briefly outlined procedure step by step using the Support Vector Regression.

## Grid Search vs. Bayesian Optimization <a name="hyperparameter_comparison"/>

In order to be able to explain the just described concept step by step with a more realistic example, I am using the Boston Housing Data and utilize the support vector
regression algorithm to build a model which approximates the correlation between:

**Target variable:** MEDV - Median value of owner-occupied homes in $1000's

**Independent variable:** LSTAT - % lower status of the population

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/boston_housing_data_set.png" style="width:100%">
  <figcaption align = "center">
    <b>Boston housing data set - Image by the author (Data: [CST79])</b>
  </figcaption>
</figure>

The aim is thus to find the hyperparameter settings for which the resulting regression model shows the best possible representation of the data set at hand.

### Support Vector Regression - How it works

In order to be able to understand the following hyperparameter optimization steps,
I will briefly describe the Support Vector Regression, how it works and the associated hyperparameters. If you are familiar with the
Support Vector Regression feel free to skip the following section.

The $$P_{xi}$$ functionality of the Support Vector Regression (SVR) is based on the Support Vector Machine (SVM). Basically, we are looking for the linear function:

$$f(x)=\langle w, x \rangle + b$$

⟨w, x⟩ describes the cross product. The goal of SV Regression is to find a straight line as model for the data points whereas
the parameters of the straight line should be defined in such a way that the line is as ‘flat’ as possible. This can be achieved
by minimizing the norm: [Wei18][Ber85]

$$\| w \|_2 := \sqrt{ ( w_1 )^2 + ( w_2 )^2 + \dotsb + ( w_n )^2 } = \left( \sum_{i=1}^n ( w_i )^2 \right)^{1/2} \label{eq:norm}$$

For the model building process it does not matter how far the data points are from the modelled straight line as long as they are
within a defined range (-ϵ to +ϵ). Deviations that exceed the specified limit ϵ are not allowed.

$$
\begin{align*}
  & minimize \quad \frac{1}{2} \cdot \| w \|^2 \\
  & subject~to~
      \begin{cases}
      y_i - \langle w, x_i \rangle -b<=\epsilon \\
      \langle w, x_i \rangle +b-y_i<=\epsilon
      \end{cases}
\end{align*}
$$

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/function_of_support_vector_regression_slack_variable.png" style="width:100%">
  <figcaption align = "center">
    <b>The soft margin loss setting for a linear SVM — Image by the author (inspired by [Smo04])</b>
  </figcaption>
</figure>

The figure above describes the "punishment" of deviations exceeding the amount of ϵ using a linear loss function. The loss function is called the kernel. Besides the
linear kernel, the polynomial or RBF kernel are frequently in use. [Smo04][Yu12][Bur98] Thus, the formulation according to Vapnik is as follows:

$$
\begin{align*}
  & minimize \quad \frac{1}{2} \cdot \| w \|^2 + C \sum_{i=1}^l (\zeta_i+\zeta_i^*) \\
  & subject~to~
      \begin{cases}
      y_i - \langle w, x_i \rangle -b<=\epsilon + \zeta_i \\
      y_i - \langle w, x_i \rangle -b<=\epsilon + \zeta_i \\
      \zeta_i,\zeta_i^*<=0
      \end{cases}
\end{align*}
$$

### Model Performance Evaluation

To evaluate the performance of the model for various hyperparameter settings, a
suitable loss function needs to be defined. An often used cost function $$L(f,x,y)$$ for
regression problems is the **Mean Squared Error (MSE)**:

$$
L(f,x,y) = MSE = \frac{1}{n}\sum_{i=1}^n(f(x)-y)^2
$$

where $$f = A(D)$$ represents the function/model returned by the algorithm $$A$$ when trained on the trainings data set $$D=z_1,z_2,...,z_k$$. $$z$$ describes the instances of the training data set, which was used to train the model $$f = A(D)$$. The variable $$x$$ describes the hyperparameters.
$$y$$ are the observed values of the variable being predicted. In terms of hyperparameter optimization, the observed values are the calculated
losses (e.g. the Mean Squared Error).


The performance of the machine learning estimator depends on the hyperparameters and the dataset used for training and validation. That's the reason, why
we are usually not just choosing a part of the data set as trainings set and another as test set and calculating the MSE for each observation of the
test data set.

To at least mitigate this effect on the performance assessment and get a more generalized assessment, the statistical procedure K-fold cross-validation (CV)
is used in the following.

Therefore, the data set is split in $$K$$ subsets. Afterwards, $$K-1$$ subsets are used as training data set, one for validation. After the model was build,
the MSE for the validation data set is calculated. This procedure is repeated until each subset has been used once as a validation data set.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/cross_validation_explained.png" style="width:100%">
  <figcaption align = "center">
    <b>Cross validation explained - Image by the author</b>
  </figcaption>
</figure>

Thus, K-models are formed and evaluated in the course of the cross validation. The cross validation score is usually calculated
as an average value from the individual Mean Square Errors.

$$CV(D, \lambda) = \frac{1}{k}\sum_{k=1}^K \frac{1}{m} \sum_{z_i \in T_k}L(A_{\lambda}(D_k), z_i)$$

If you use the sklearn module **sklearn.model_selection.cross_val_score** and want to use the MSE as the scoring parameter,
you will notice that only the negated MSE can be selected. This is due to the uniform convention of trying to maximize scores in all
cases. Therefore, cost functions are always negated.

<script src="https://gist.github.com/polzerdo55862/ac8cd911802b574693cbe5aa1247c837.js"></script>

The function of the negative cross validation score thus represents the **Objective Function** of the mathematical optimization problem.
The objective is to identify the optimal hyperparameter settings, i.e. the hyperparameter values for which the trained models show
the best performance (i.e. the negative cross validation score is maximal).

**Objective function:** The objective function of a mathematical optimization problems is the real-valued function which should be minimized or maximized.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_function_evaluation.png" style="width:100%">
  <figcaption align = "center">
    <b>Optimization problem- Image by the author</b>
  </figcaption>
</figure>

Since we have no knowledge about the analytical form of f(x) at the first moment, we speak about a so-called black-box function. A black-box function is a system where the internal workings are unknown.
Systems like transistors, engines and human brains are often described as black-box systems.

In our case, the hyperparameters represent the input parameters of the function, for which only it is not directly known how it influences e.g. the cross validation score.
Each point on the function must be calculated more or less elaborately.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_function.png" style="width:100%">
  <figcaption align = "center">
    <b>Black-box function – Image by the author</b>
  </figcaption>
</figure>

### Finding the optimal hyperparameter settings

In order to find the optimal hyperparameter settings, we could theoretically calculate the cross validation score
for each possible hyperparameter combination, and finally choose the hyperparameters that show the best performance in the evaluation.

The following figure shows the procedure for the hyperparameter space $$Epsilon = 0.1-16$$, where $$C$$ constantly takes the value $$7$$.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_calculation.png" style="width:100%">
  <figcaption align = "center">
    <b>Sample calculation of the black-box function for different hyperparameter settings – Image by the author</b>
  </figcaption>
</figure>

The hyperparameter optimization process "Grid Search" works according to this procedure. We thus build up the function approximately with each calculation step, bit by bit.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/Build_2d_evaluation.gif" style="width:100%">
  <figcaption align = "center">
    <b>Build up black-box function step for step – Image by the author</b>
  </figcaption>
</figure>

How close we actually come to the optimum in the area under consideration thus depends decisively on the step size (the fineness of the net).
However, if we choose a very small step size, have numerous hyperparameters,
a large data set and possibly also use an algorithm that functions according to a relatively
computationally intensive principle, the computational effort required for the search for the
optimal hyperparameters could increase rapidly.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/2d_evaluation_various_step_sizes.gif" style="width:100%">
  <figcaption align = "center">
    <b>Calculated black-box using various step sizes – Image by the author</b>
  </figcaption>
</figure>


### Grid Search <a name="grid search"/>

As shown in the following figure, we define a "grid" over the hyperparameter space. If we consider the kernel to be fixed
for the moment, the following two-dimensional hyperparameter space results.
($$C_{min} = 1$$, $$C_{max} = 50$$, $$\epsilon_{min} = 1$$, $$\epsilon_{max} = 30$$, $$step\_size = 1$$)

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/grid_search_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Grid search illustration – Image by the author</b>
  </figcaption>
</figure>

The figure already shows that the optimum in the selected hyperparameter space must lie approximately in the lower,
right-hand part.

The dots in the right-hand graph indicate which hyperparameter combination was investigated. For the example shown, 1500 hyperparameter combinations were evaluated.
Since we use cross validation for evaluation,
a 5-fold cross validation results in $$1500~x~5 = 7500$$ models that have to be built and evaluated.

Although we choose an exceptionally high granularity for the example, algorithms with 3, 4 or 5 hyperparameters
nevertheless, require an enormous amount of computing power.

### From Grid to Bayesian Optimization <a name="grid to baysian"/>

Basically a valid approach, but if one is talking about so-called "costly" black-box functions,
it is worthwhile to use alternative hyperparameter optimization methods [Cas13].

The distinction between "costly" and "cheap" is usually made on the basis of the time required for the evaluation,
the computing power required and/or the capital investment required.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/cheap_and_costly_black_box_function.png" style="width:100%">
  <figcaption align = "center">
    <b>Cheap vs. costly black-box functions – Image by the author</b>
  </figcaption>
</figure>

The computational effort required to calculate the black box function depends on various factors, such as the dimensionality
of the hyperparameter space, the way the chosen algorithm works, as well as the subsequent evaluation of
the models formed.

If the available computing power is limited and a hyperparameter combination
already takes several seconds or minutes, it may make sense to look for solutions that reduce
the number of required data points.

Bayesian Optimization introduces the Surrogate Function for this very purpose. In this case,
the Surrogate Function is a calculated regression model that is supposed to approximate the real
black-box function on the basis of a few sampling points.

Basically, in Bayesian Optimization we try to reduce the uncertainty of the model step by step, with each additional
sampling point calculated - usually focusing on areas where the global maximum of the function is likely to lie.

Sounds like an extremely effective approach in itself, although one must take into account that this procedure also
results in additional computational effort and that a sufficient replication of the black-box function cannot always
be achieved. As with any other regression problem, the formation of a sufficiently good model cannot be taken for granted.
While Grid Search ends with the evaluation of the model performance, Bayesian hyperparameter optimization
additionally calculates the Surrogate and Acquisition Function.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/evaluation_steps.png" style="width:100%">
  <figcaption align = "center">
    <b>Evaluation steps: Grid Search vs. Baysian Optimization – Image by the author</b>
  </figcaption>
</figure>

#### Surrogate Function - the Gaussian Process Regression

As described above, the aim is to find a Surrogate Function which approximates the black-box function as close
as possible by using less calculated points.

The best-known Surrogate Function in the context
of hyperparameter optimization is the Gaussian process, or more
precisely the Gaussian process regression. A more detailed explanation
of how Gaussian Process Regression works can be found in "Gaussian
Processes for Machine Learning" by Carl Edward Rasmussen and Christopher
K. I. Williams, which is available for free at:

<a href="http://www.gaussianprocess.org/gpml/chapters/">www.gaussianprocess.org</a>

Or take a look at one of my previous articles describing how GP regression works:

<a href="https://towardsdatascience.com/7-of-the-most-commonly-used-regression-algorithms-and-how-to-choose-the-right-one-fc3c8890f9e3
">towardsdatascience.com/7-of-the-most-commonly-used-regression-algorithms-and-how-to-choose-the-right-one</a>

In short, Gaussian process regression defines a priori Gaussian process that already
includes prior knowledge of the
true function. Training on a given data set results in the Priori Gaussian Process.

In order to calculate a first posteriori Gaussian process, we need a calculated sample point of the true black-box function. Using this
calculated "support point", we can already build a first GP regression model.

<script src="https://gist.github.com/polzerdo55862/24a79ea4467e7ff7000cbeca37c5c5c6.js"></script>

Since we usually have no prior knowledge about how the black box function looks like, we choose for the Priori Gaussian Process a mean function which is
a parallel straight line to the x-axis ($$y=0$$). As a kernel, we use one of the most frequently used kernels, the Radial Basis Function (RBF).
Since we assume that the point (here in red) is part of the "true" black box function by directly calculating the sampling points
, a covariance of zero results at the position of the calculated point. In the figure, the level of model
uncertainty is visualized by the standard deviation and highlighted in grey. By knowing individual data points of the true function, the possible course of the
function is gradually narrowed down.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/first_gp_model.png" style="width:100%">
  <figcaption align = "center">
    <b>Posteriori Gaussian Process – Image by the author</b>
  </figcaption>
</figure>

It is precisely this measure of the uncertainty of the model that we use in the following to identify the "best possible"
next sampling point. As described briefly above, in this case we have the situation that we can freely choose the next calculation step.
For this very purpose, we are introducing a so-called Acquisition Function.


#### Acquisition Function

The most popular Acquisition Function in the context of hyperparameter
optimization is the **Expected Improvement (EI)**. Further Acquisition Functions are the "Probability of Improvement", "Knowledge Gradient", "Entropy Search" or "Predictive Entropy".

The EI is defined as following [Kra18][Jon98][Uai18][Has19]:

$$\operatorname{EI}(\mathbf{x}) =
\begin{cases}
(\mu(\mathbf{x}) - f(\mathbf{x}^+) - \xi)\Phi(Z) + \sigma(\mathbf{x})\phi(Z)  &\text{if}\ \sigma(\mathbf{x}) > 0 \\
0 & \text{if}\ \sigma(\mathbf{x}) = 0
\end{cases}$$

where

$$
Z =
\begin{cases}
\frac{\mu(\mathbf{x}) - f(\mathbf{x}^+) - \xi}{\sigma(\mathbf{x})} &\text{if}\ \sigma(\mathbf{x}) > 0 \\
0 & \text{if}\ \sigma(\mathbf{x}) = 0 \end{cases}
$$

and

* $$\mu$$: is the mean of the distribution defined by the Gaussian process
* $$\sigma$$: is the standard deviation of the distribution defined by the Gaussian process
* $$\Phi()$$: is the standard normal cumulative density function (cdf)
* $$\phi()$$: is the standard normal probability density function (pdf)
* $$\xi$$: is an exploration parameter

For the illustrative example, in the first step, we choose a random sampling set for which we determine the value of the black-box function.
For this example we select only one sampling point for the first step and fit a first Gaussian process regression model to it.
Since we assume no noise, the covariance in the area of the sampling point becomes zero,
the mean of the regression line runs directly through the point. To the right and left of this point, the covariance increases, and with it the uncertainty of the model.

With the above formula, we now calculate the acquisition function for the hyperparameter space.
$$f(x^{+})$$ describes the maximum value of all sample points calculated so far. Since we have only calculated one point in the figure,
$$f(x^{+})$$ is the function value at the selected sample point - here at $$-31.0$$. $$\sigma$$ and $$\mu$$ are described by the Gaussian
process regression model. In the figure, you can see both values for the position $$x=15$$.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/expected_improvement_explained.png" style="width:100%">
  <figcaption align = "center">
    <b>Posteriori Gaussian Process – Image by the author</b>
  </figcaption>
</figure>

If we now look at the formula more closely, we notice that EI consists of two parts:

* the left part describes the difference between the mean value of the Gaussian process regression model and the max. f(x) value of all sampling points
* the right part, the uncertainty of the model using the standard deviation

How both parts are weighted depends on CDF(Z) and PDF(Z). If the difference between $$f(x^{+})$$ and $$\mu$$ is large compared to the standard deviation of the regression model,
CDF(Z) goes towards 1 and PDF(Z) towards 0. This means that areas are weighted more heavily where the mean value of the model is significantly higher than the maximum $$f(x^{+})$$ of
the sampling points so far. The exploration parameter can be set freely and thus control the weighting somewhat.

One speaks here of a tradeoff between **exploration** and **exploitation**, which is reflected by the two term components:

1. Explorative: select the point on the function where the current model shows the greatest uncertainty (explore other parts of the search space with the hope to find other promising areas [Leh13])
2. Exploitative: select the point that now shows the greatest value and explore the area more closely

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/expected_improvement_explained_2.png" style="width:100%">
  <figcaption align = "center">
    <b>Posterior Gaussian Process – Image by the author</b>
  </figcaption>
</figure>

After calculating the acquisition function, we simply identify the hyperparameter value $$x$$ (here: Epsilon) at which EI is maximal and perform the black-box function calculation for this value.
Then we perform the calculation of the acquisition function again and identify the next sampling point for the next iteration.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/Visualize_baysian_opt.gif" style="width:100%">
  <figcaption align = "center">
    <b>Bayesian Optimization step by step – Image by the author</b>
  </figcaption>
</figure>

## Summary

The article deals with Bayesian hyperparameter optimization and explains how it can help to find the optimal hyperparameter settings more efficiently by reducing the required sample points and thus the computational effort.

The article does not claim to represent a full picture of Bayesian optimization. Neither the practical application with libraries such as **scikit-optimizer** or **hyperopt**, nor the comparison of different optimisation methods on concrete application examples are dealt with.

The aim of the article is to present the basic functioning of Bayesian hyperparameter optimization as simply and comprehensibly as possible. In doing so, the ar ticle highlights how Bayesian hyperparameter optimization differs from other methods such as grid search.

If you liked the article, feel free to check out one of my others that explains how various regression techniques, anomaly detection methods, …:

If you are not yet a Medium Premium member and plan to be after reading this article, you can support me by signing up via the following referral link:

Thank you for reading!

## References

[Agn20] Agnihotri, Apoorv; Batra, Nipun. Exploring Bayesian Optimization. https://distill.pub/2020/bayesian-optimization/. 2020. <br>
[Bur98] Burges, C. J. C.; Kaufman, L.; Smola, A. J.; Vapnik, V. Support Vector Regression Machines. 1998. URL http://papers.nips.cc/paper/1238-support-vector-regression-machines.pdfine <br>
[Cas13] Cassilo, Andrea. A Tutorial on Black–Box Optimization. https://www.lix.polytechnique.fr/~dambrosio/blackbox_material/Cassioli_1.pdf. 2013.<br>
[CST79] U.S. Census Service. https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html <br>
[Fah16] Fahrmeir, L.; Heumann, C.; Künstler, R. Statistik: Der Weg zur Datenanalyse. Springer-Lehrbuch. Springer Spektrum, Berlin and Heidelberg, 8., überarbeitete und ergänzte auflage Auflage, 2016. ISBN 978–3–662 50371–3. doi:10.1007/978–3–662–50372–0 <br>
[Has19] Bayesian Hyperparameter Optimization using Gaussian Processes. 2018. https://brendanhasz.github.io/2019/03/28/hyperparameter-optimization.html#hyperparameter-optimization <br>
[Hel19] Helmenstine, Anne Marie. Bayes Theorem Definition and Examples. https://www.thoughtco.com/bayes-theorem-4155845. 2019. <br>
[Jon98] Jones, D.R., Schonlau, M. & Welch, W.J. Efficient Global Optimization of Expensive Black-Box Functions. Journal of Global Optimization <br>
[Kra18] Martin Krasser. Bayesian optimization. http://krasserm.github.io/2018/03/21/bayesian-optimization/ <br>
[Leh13] H.E. Lehtihet. https://www.researchgate.net/post/What_is_the_difference_between_exploration_vs_exploitation_intensification_vs_diversification_and_global_search_vs_local_search
[Sci18] Sicotte, Xavier. Cross validation estimator. https://stats.stackexchange.com/questions/365224/cross-validation-and-confidence-interval-of-the-true-error/365231#365231. 2018 <br>
[Smo04] Smola, A. J.; Schölkopf, B. A tutorial on support vector regression. Statistics and Computing, 14(3):199–222, 2004. ISSN 0960–3174. doi:10.1023/B:STCO. 0000035301.49549.8849549. <br>
[Uai18] UAI 2018. https://www.youtube.com/watch?v=C5nqEHpdyoE. 2018. <br>
[Yu12] Yu, H.; Kim, S. SVM Tutorial — Classification, Regression and Ranking. G. Rozenberg; T. Bäck; J. N. Kok, Handbook of Natural Computing, 479–506. Springer Berlin Heidelberg, Berlin, Heidelberg, 2012. ISBN 978–3–540–92909–3. <br>
[Was21] University of Washington. https://sites.math.washington.edu/. 2021. <br>
