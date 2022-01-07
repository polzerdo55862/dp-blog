---
title: Baysian Optimization

author:
  name: Dominik Polzer
  link: https://github.com/polzerdo55862
date: 2022-01-05 18:32:00 -0500
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
---

# Bayesian-Hyperparameter-Optimization

### Table of content

* [Introduction](#introduction)
* [Hyperparameter Optimization](#hyperparameter opt)
    * [Objective Function](#objective)
    * [Grid Search](#grid search)
    * [From Grid Search to Baysian Optimization](#baysian)
    * [Baysian Optimization](#baysian opt)



## Introduction <a name="introduction"/>


The performance of a machine learning method depends on the used data set and the
chosen hyperparameter settings for the model training.
Finding the optimal hyperparameter settings is crucial for building the best possible model
for the given data set.

This article describes the basic procedure for hyperparameter optimisation. Simple procedures such
as grid search, which scan a defined hyperparameter space, reach their limits when the calculation
of the loss function is computationally intensive.

In order to reduce the computing power required to find the optimal hyperparameter settings,
Baysian optimisation uses Bayes' theorem.

In simple terms, the Bayes' theorem is used to calculate the probability of an event based on its
association with another event [Hel19].

<img src="/img/bayes_theorem.png" style="width:100%">
<figcaption align = "center">
    <b>Bayes' theorem</b>
</figcaption>

A popular application ot the Bayes' theorem is the diesease detection. For rapid tests one is interested in how high the actual probability is,
that a positive tested person actually has the diseas.[Fah16]

In the context of hyperparameter optimisation, we would like to infer the probability distribution of the loss values of a second non-calculated
hyperparameter combination by calculating the loss of a hyperparameter combination.

With the help of some calculated "true" values of the loss function, we would like to model the function of
the loss function over the entire hyperparameter space - a so-called surrogate function.

In Gauss Process Regression, the resulting model provides not only an approximation of the true loss function but also the confidence interval to the non-calculated function areas.
This information is used in the following to identify the hyperparameter combination whose calculation brings the greatest "information gain",
which will probably help the most to model an accurate Surrogate Function.

In the following, the Boston Housing data set (https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) is used to illustrate the procedure for
hyperparameter optimization. More specifically, we want to model a possible correlation between MEDV and LSTAT.

**Target varibale:** MEDV - Median value of owner-occupied homes in $1000's

**Independend variable:** LSTAT - % lower status of the population

<img src="/img/boston_housing_data_set.png" style="width:100%">
<figcaption align = "center"><b>Boston housing data set - Image by the author (Data: [CST79])</b></figcaption>

In the following, we want to use Support Vector Regression (SVR) for model building.

## Support Vector Regression

The functionality of the Support Vector Regression (SVR) is based on the Support Vector Machine (SVM)
and will first be explained with a simple example. We are looking for the linear function:

<img src="/img/function_of_support_vector_regression_slack_variable.png" style="width:100%">
<figcaption align = "center"><b>Boston housing data set - Image by the author (Data: [CST79])</b></figcaption>

To evaluate the performance of the model for various hyperparameter settings a
suitable loss function needs to be defined. An often used cost function for
regression problems is the Mean Squared Error (MSE):


<img src="/img/loss_function.png" style="width:100%">
<figcaption align = "center"><b>Loss Function: Mean Squared Error (MSE) - Image by the author</b></figcaption>

The performance of the machine learning estimator depends on the hyperparameters and the dataset used for training and validation. In order to obtain a generalised assessment of the performance of the algorithm used, the statistical procedure k-fold cross-validation (CV) is used in the following.

Therefor the data set is split in k subsets. Following k-1 subsets are used as training data set, one for validation. After the model was build, the MSE for the validation data set is calculated. This procedure is repeated until each subset has been used once as a validation data set. The CV score is then calculated as the average of the MSE values.

<img src="/img/cross_val_score.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

The target of hyperparameter optimization is to find the optimal hyperparameter settings,
in this case, where the Loss (e.g. the CV Score) is minimal (in the following we try find
the maximum for the negative CV score).

* **Objective function:** The objective function of a mathematical optimization problems is the real-valued function which should be minimized or maximized []
<img src="/img/black_box_function_evaluation.png" style="width:100%">
<figcaption align = "center"><b>Optimization problem- Image by the author"</b></figcaption>

* **Black-box function:** A black-box function is a system where the internal workings is unknown. Systems like transistors, engines and human brains are often described as black-box systems.
<img src="/img/black_box_function.png" style="width:100%">
<figcaption align = "center"><b>Black-box function – Image by the author (inspired by [Sic18])</b></figcaption>

What we could do, is just calculate the value of f(x) for mupltiple hyperparameter settings
in the defined hyperparameter space and choose the hyperparameter combination with the lowest
loss - like grid is doing it.

<img src="/img/black_box_calculation.png" style="width:100%">
<figcaption align = "center"><b>Sample calculation of the black-box function for different hyperparameter settings – Image by the Author</b></figcaption>

## Grid Search <a name="grid search"/>

Grid Search is the easiest way to search for the optimal Hyperparameter settings.

1. Choose an appropriate scale of your hypparameters and define a search space
2. Iterate in steps over the defined space and calculate the value of te black-box function - here the cross-val. score

<img src="/img/grid_search_example.png" style="width:100%">
<figcaption align = "center"><b>s – Image by the Author</b></figcaption>

# From Grid to Baysian Optimization <a name="grid to baysian"/>

Definitely a valid approach, at least for so called “cheap” black-box function, where the computation effort to calculate the CV values is low.
But what if the evaluation of the function is costly, so the computational time and/or cost to calculate CV is high?
In this case it may makes sense to think about more “intelligent” ways to find the optimal value. [Cas13]

<img src="/img/cheap_and_costly_black_box_function.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

One way is to define a "cheap" Surrogate Function.
The Surrogate Function should approximates the black-box function f(x) [Cas13].

Similiar to most regression problems, we want to model a surrogate function of the black-box function
using a few calculated values, that gives a prediction for the hyperparameter space.

To model the surrogate function, a wide range of machine learning techniques is used,
like Polynomial Regression, Support Vector Machine, Neuronal Nets and probably the most
popular, the Gaussian Process (GP).

Bayesian optimisation can thus be assigned to the field of active learning. Active Learning
tries to mimize the labelling costs.
The aim is to replicate the black-box function as accurately as possible with as
little computational effort as possible.
computational effort.

If we speak of Gaussian hyperparameter optimisation, we are moving in the
field of uncertainty reduction.

As a rule, the variance is used as a measure of uncertainty. The Gaussian Process (GP)
is able to map the
the uncertainty as well. [Agn20]

For the above regression problem, the following black-box function results.
In order to be able to map the function with sufficient accuracy for the defined
hyperparameter space, this range must be appropriately fine-granularly ebased.
In this case we assume a predefined hyperparameter space (epsilon = 1 - 15).

<img src="/img/hyperparameter_evaluation_2d_gif_step_size_1.1.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

In total, the time needed to compute the needed sample values and the surrogate function,
should be less time-consuming than calculating each point in the hyperparameter space.

<img src="/img/evaluation_steps.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

### Surrogate Function - the Gaussian Process Regression

As described above, the aim is to find Surrogate Function which approx. the black-box function as close
as possible (or necessary) by using less calculated points.

 The best-known surrogate function in the context
of hyperparameter optimisation is the Gaussian process, or more
precisely the Gaussian process regression. A more detailed explanation
of how the Gaussian Process Regression works can be found in "Gaussian
Processes for Machine Learning" by Carl Edward Rasmussen and Christopher
K. I. Williams, which is available for free at:

http://www.gaussianprocess.org/gpml/chapters/

You can also find an explanation of Gauss Process Regression in one of my recent articles:

https://towardsdatascience.com/7-of-the-most-commonly-used-regression-algorithms-and-how-to-choose-the-right-one-fc3c8890f9e3

In short, Gaussian process regression defines a priori Gaussian process that already
includes prior knowledge of the
true function. Since we usually have no knowledge about the true course of our
black box function, a constant function
with some covariance is usually freely chosen as the Priori Gauss.


....

By knowing individual data points of the true function, the possible course of the
function is gradually narrowed down.


### Acquisition Function

The surrogate function is recalculated after each calculation step and serves as the
basis for selecting the next calculation step.
For this purpose, an acquisition function is introduced. The most popular
acquisition function in the context of Hpyer parameter
optimisation is the information gain.

In addition to the Expected Improvement the following Acquisition Functions are used:

- Knowledge gradient
- Entropy search
- Predictive entropy


<img src="/img/baysian_optimization_plots.png" style="width:100%">
<figcaption align = "center"><b>K-fold Cross Validation – Image by the author (inspired by [Sic18])"</b></figcaption>

## References

[Agn20] Agnihotri, Apoorv; Batra, Nipun.  https://distill.pub/2020/bayesian-optimization/. 2020. <br>
[Cas13] Cassilo, Andrea. A Tutorial on Black–Box Optimization. https://www.lix.polytechnique.fr/~dambrosio/blackbox_material/Cassioli_1.pdf. 2013.<br>
[CST79] U.S. Census Service. https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html <br>
[Fah16] Fahrmeir, L.; Heumann, C.; Künstler, R. Statistik: Der Weg zur Datenanalyse. Springer-Lehrbuch. Springer Spektrum, Berlin and Heidelberg, 8., überarbeitete und ergänzte auflage Auflage, 2016. ISBN 978–3–662 50371–3. doi:10.1007/978–3–662–50372–0
[Hel19] Helmenstine, Anne Marie. Bayes Theorem Definition and Examples. https://www.thoughtco.com/bayes-theorem-4155845. 2019. <br>
[Sci18] Sicotte, Xavier. Cross validation estimator. https://stats.stackexchange.com/questions/365224/cross-validation-and-confidence-interval-of-the-true-error/365231#365231. 2018 <br>
[Was21] University of Washington. https://sites.math.washington.edu/. 2021. <br>

