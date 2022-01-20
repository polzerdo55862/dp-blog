---
title: Bayesian Hyperparameter Optimization

author:
  name: Dominik Polzer
  link: https://github.com/polzerdo55862
date: 2022-01-05 18:32:00 -0500
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
---

### Table of content

* [Introduction](#introduction)
* [Hyperparameter Optimization](#hyperparameter opt)
    * [Objective Function](#objective)
    * [Grid Search](#grid search)
    * [From Grid Search to Baysian Optimization](#baysian)
    * [Baysian Optimization](#baysian opt)


## Introduction <a name="introduction"/>

Hyperparameters are parameters that are set before the actual training process to control the learning process. For the decision tree we limit the maximum number of nodes
(the maximum depth of the tree); for polynomial regression the polynomial degree; for support vector regression the kernel,
the regularisation parameter C and the margin of tolerance Epsilon. All these parameters influence the training process and thus the performance of the resulting model.

The search for optimal hyperparameters is called Hyperparameter Optimization. The search is for the hyperparameter combination for which the trained model shows
the best performance for the given data set. Popular methods for this are Grid Search, Random Search and Baysian Optimisation. I will explain the exact difference
in the approach in the course of the article. The decisive factor for choosing the right optimisation method is the computational effort required to evaluate the
various different hyperparameter settings.

If we want to know how the resulting model performs for a specific hyperparameter combination, we have no choice but to build the model using one subset of the dataset and then evaluate it using a second subset.
build the model using a subset of the data set and then evaluate it using a second subset. The functioning of the algorithm, the
selected hyperparameter settings and the size of the dataset determine to a large extent how computationally expensive the model building is. Especially with very
It is worth going beyond the principle of "simple" trial and error, especially in the case of very computationally intensive model-building processes.

For illustrative purposes, consider a simple regression problem that we would like to solve using polynomial regression. As already mentioned in the first step we
choose settings for the hyperparameters that seem plausible to us (e.g. based on prior knowledge). Most problems from fields such as engineering or physics, for
example, can usually be solved sufficiently well with polynomial regression models with polynomial degrees of less than 10. can be sufficiently well explained.

For the following evaluation we set the hyperparameter Space to polynomial degrees between 0 - 20. To ensure that we identify the optimal hyperparameter value in
the defined hyperparameter space, we could build a model for each polynomial degree and evaluate it. In regression, we usually compare the absolute or squared deviation
of the test values from the predicted values of the model. For more details I explain this procedure below.

With a for-loop, we could simply perform the following steps for each possible hyperparameter and cache the results of each run:

1. Define the polynomial regression algorithm with the specified hyperparameter setting
2. Build the model (using the train data set)
3. Evaluate the model (using the validation or test data set)

Afterwards we simply choose the polynomial degree for our model that shows the best performance in the evaluation process.

[comment]: <> (Hyperparameter sind parameter welche vor dem eigentlich Trainingsprozess festgelegt werden um den Lernprozess zu steuern. Beim Decision Tree begrenzen wir die)

[comment]: <> (die maximale Anzahl der nodex &#40;die maximale Tiefe des Baums&#41;; bei der Polynomial Regression den Polynomgrad; bei der Support Vector Regression den Kernel, den Regularization)

[comment]: <> (Parameter C und die margin of tolerance Epsilon. All diese Parameter beeinflussen den Trainingsprozess und damit die Performance des resultierenden Models.)

[comment]: <> (Die Suche nach optimalen Hyperparametern is called Hyperparameter Optimization. Gesucht wird die Hyperparameter Kombination für die das trainierte model die beste)

[comment]: <> (performance für den given data set zeigt. Populäre Methode hierfür sind Grid Search, Random Search and Baysian Optimization. Den genauen Unterschied in der)

[comment]: <> (Vorgehensweise erläutere ich im Laufe des Artikels. Der entscheidende Faktor, für die Wahl des richtigen Optimierungsverfahren, ist der Rechenaufwand zur evaluation)

[comment]: <> (verschiedenster Hyperparameter Settings.)

[comment]: <> (Möchten wir zu einer speziellen Hyperparameter Kombination wissen, wie das resultierende Model performt, bleibt uns im ersten Moment nichts anderes übrig, als)

[comment]: <> (das Model mithilfe eines Subsets des Datensatzes zu bilden und anschließend anhand eines zweiten Subsets zu evaluieren. Die Funktionsweise des Algorithmus, die)

[comment]: <> (gewählten Hyperparameter Settings und die größe des Datensatzes bestimmen dabei maßgebend, wie Rechenaufwendig die Modelbildung ist. Gerade bei sehr)

[comment]: <> (rechenaufwendigen Modellbildungsprozessen, lohnt es sich, über das Prinzip des "einfachen" Ausprobierens hinauszugehen.)

[comment]: <> (Betrachten zur Anschaulichkeit ein einfaches Regressionsproblem, welches wir mithilfe der Polynomial Regression lösen möchten. Wie bereits erwähnt)

[comment]: <> (wählen wir im ersten Schritt Settings für die Hyperparameter die uns &#40;z.B. anhand von Vorwissen&#41; als plausibel erscheinen. Die meisten Probleme aus)

[comment]: <> (Bereichen wie des Ingenieurwesens oder der Physik können beispielsweise in der Regel mit Polynomial Regressionsmodelle mit Polynomgraden von unter 10)

[comment]: <> (ausreichend gut angenehärt werden.)

[comment]: <> (Für die folgenden Evaluierung legen wir den Hyperparameter Space auf Polynomgrade zwischen 0 - 20 fest. Um sicherzugehen, das wir den optimalen)

[comment]: <> (Hyperparameter Wert im definierten Hyperparameter Space zu identifizieren, könnten wir für jeden Polynomgrad ein Model bilden uns dieses Evaluieren.)

[comment]: <> (Bei der Regression vergleichen wir in der Regel die absolute oder quadratische Abweichung der Testwerte von den predicteten Werte des Models. Genauer)

[comment]: <> (erkläre ich dieses vorgehen weiter unten.)

[comment]: <> (Mit einer for-loop könnten wir die folgenden Schritte einfach für jeden möglichen Hyperparameter ausführen und die Ergebnisse eines jeden Durchlaufes zwischenspeichern:)

[comment]: <> (1. Define the polynomial regression algorithm with the specified hyperparameter setting)

[comment]: <> (2. Build the model &#40;using the train data set&#41;)

[comment]: <> (3. Evaluate the model &#40;using the validation or test data set&#41;)

[comment]: <> (Afterwards we simply choose the polynomial degree for our model that shows the best performance in the evaluation process.)

[comment]: <> (___________________________)

[comment]: <> (The performance of a machine learning method depends massivly on chosen hyperparameter settings.)

[comment]: <> (Finding the optimal hyperparameter settings is crucial for building the best possible model)

[comment]: <> (for the given data set.)

[comment]: <> (This article describes the basic method for hyperparameter optimisation. Simple procedures such)

[comment]: <> (as grid search, which scan a defined hyperparameter space, is not very effective when the calculation)

[comment]: <> (of the loss function is computationally intensive.)

[comment]: <> (What exactly I mean by this, I will try to describe briefly. In the context of this article, I use regression as an example to)

[comment]: <> (illustrate the procedure for finding the optimal hyperparameter settings. To estimate the performance of)

[comment]: <> (the generated model, we calculate the loss between predicted and actual values of a test training set that was not used to train the model.)

[comment]: <> (For example, if we choose polynomial regression as the regression algorithm, we have the possibility to adjust the model complexity)

[comment]: <> (by choosing the polynomial degree. Thus, the polynomial degree represents a hyperparameter of the polynomial regression.)

[comment]: <> (In order to find the polynomial degree that best reproduces the complexity of the problem and data set at hand, we could let the)

[comment]: <> (polynomial degree take on any conceivable value, calculate the performance of the generated model via the loss and then choose)

[comment]: <> (the polynomial degree at which the resulting model has shown the lowest loss.)

[comment]: <> (But which values are conceivable for the polynomial degree? - The polynomial degree can basically be any integer value. From experience,)

[comment]: <> (it can be said that most problems in fields such as engineering can be represented sufficiently accurately by models with polynomial degrees of less than 10.)

[comment]: <> (Based on this experience, we could limit the hyperparameter space, for example, to a polynomial degree between 1 and 20.)

[comment]: <> (In order to be sure to identify the polynomial degree at which the model shows the best performance,)

[comment]: <> (we would have to evaluate every possible hyperparameter setting in this defined hyperparameter space.)

[comment]: <> (We could implement a for-loop that performs the following calculation steps for each integer value between 0 and 20:)

[comment]: <> (1. Define the polynomial regression algorithm with the specified hyperparameter setting)

[comment]: <> (2. Build the model &#40;using the train data set&#41;)

[comment]: <> (3. Evaluate the model &#40;using the validation or test data set&#41;)

[comment]: <> (Afterwards we simply choose the polynomial degree for our model that shows the best performance in the evaluation process.)

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/grid_search_polynomial_regression_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Hyperparameter search: simple for-loop - Image by the author</b>
  </figcaption>
</figure>

This approach is certainly valid for the dataset shown and relatively simple polynomial regression models, but reaches its
limits when the datasets and hyperparameter space are large and complex and more computationally expensive algorithms are used.

In order to reduce the computing power required to find the optimal hyperparameter settings,
Baysian optimisation uses Bayes' theorem. In simple terms, the Bayes' theorem is used to calculate the probability of an event, based on its
association with another event [Hel19].

So if we know the probability of observing the event $$A$$ and $$B$$ independently from each other
(the so called priori probability) and the probability of event $$B$$ occuring given that $$A$$ is true (the so called conditional probability) we are able
to calculate the probability of event $$A$$ occuring given that $$B$$ is true (conditional probability) as follows:

$$P(A| B) = \frac{P(B |  A) \cdot P(A)}{P(B)}$$

A popular application ot the Bayes' theorem is the diesease detection. For rapid tests one is interested in how high the actual probability is,
that a positive tested person actually has the diseas.[Fah16]

In the context of hyperparameter optimisation, we would like to predict the probability distribution of the loss value for any possible hyperparameter
combinitation in the defined hyperparameter space. With the help of some calculated "true" values of the loss function, we would like to model the function of
the loss function over the entire hyperparameter space - a so-called surrogate function. In our example, we could calculate the resulting loss for a polynomial
degree of 2, 8 and 16 and use regression analysis to generate a function that approximates the loss over the entire hyperparameter space from 1 to 20.

In Gaussian Process Regression, the resulting model provides not only an approximation of the true loss function but also a meassurement of the
uncertainty of the model for each hyperparameter combination. Simply put, the smaller the confidence interval (here: grey background) at a point in the function, the more certain the
model is that the mean value (here: black curve) predicts/approximates the real loss value. If one would like to increase the accuracy of the approximation of the
regression function, we could simply increase the size of the training data set. Since we can specifically choose a some hyperparameters and calculate the resulting loss of the model,
its worth to think first what sampling point would probably result in the highest model improvement. A popular way to find the next sampling point, is to use the use the
uncertainty of the model as as basis for the decision. So we would simply identify the part of model with the largest confidence interval.

These and similar considerations are mapped into an acquisition function, which is the basis for choosing the next sampling point.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/gaussian_process_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Example posteriori gaussian process - Image by the author</b>
  </figcaption>
</figure>

## Grid Search vs. Baysian Optimization using SVR

In order to be able to explain this concept step by step with a more realistic example, I am using the Boston Housing Data and utilize the support vector
regression algorithm to build a model which approximates the correlation between:

**Target varibale:** MEDV - Median value of owner-occupied homes in $1000's

**Independend variable:** LSTAT - % lower status of the population

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/boston_housing_data_set.png" style="width:100%">
  <figcaption align = "center">
    <b>Boston housing data set - Image by the author (Data: [CST79])</b>
  </figcaption>
</figure>

The aim is thus to find the hyperparameter settings for which the resulting regression model shows the best possible representation of the data set at hand
(the loss compared to the test data set becomes minimal).

### Build a first Support Vector Regression model

In order to be able to understand the following hyperparameter optimisation steps,
I will briefly describe the Support Vector Regression, how it works and the associated hyperparameters. If you are familiar with the
Support Vector Regression feel free to skip the following section.

The $$P_{xi}$$ functionality of the Support Vector Regression (SVR) is based on the Support Vector Machine (SVM). Basically we are looking for the linear function:

$$f(x)=\langle w, x \rangle + b$$

⟨w, x⟩ describes the cross product. The goal of SV Regression is to find a straight line as model for the data points whereas
the parameters of the straight line should be defined in such a way that the line is as ‘flat’ as possible. This can be achieved
by minimizing the norm: [Wei18][Ber85]

$$\| w \|_2 := \sqrt{ ( w_1 )^2 + ( w_2 )^2 + \dotsb + ( w_n )^2 } = \left( \sum_{i=1}^n ( w_i )^2 \right)^{1/2} \label{eq:norm}$$

For the model building process it does not matter how far the data points are from the modeled straight line as long as they are
within a defined range (-ϵ to +ϵ). Deviations that exceed the specified limit ϵ are not allowed.

$$
\begin{align*}
  & minimize \quad \frac{1}{2} \cdot \| w \|^2 \\
  & subject~to~\left( \begin{array}{ccc}
      y_i - \langle w, x_i \rangle -b<=\epsilon \\
      \langle w, x_i \rangle +b-y_i<=\epsilon
    \end{array} \right)
\end{align*}
$$

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/function_of_support_vector_regression_slack_variable.png" style="width:100%">
  <figcaption align = "center">
    <b>Boston housing data set - Image by the author (Data: [CST79])</b>
  </figcaption>
</figure>

The figure above describes the “punishment ”of deviations exceeding the amount of ϵ using a linear loss function. The loss function is called the kernel. Besides the linear kernel, the polynomial or RBF kernel are frequently in use. [Smo04][Yu12][Bur98]
Thus, the formulation according to Vapnik is as follows:

$$
\begin{align*}
  & minimize \quad \frac{1}{2} \cdot \| w \|^2 + C \sum_{i=1}^l (\zeta_i+\zeta_i^*) \\
  & subject~to~\left( \begin{array}{ccc}
      \quad y_i - \langle w, x_i \rangle -b<=\epsilon + \zeta_i \\
      y_i - \langle w, x_i \rangle -b<=\epsilon + \zeta_i \\
      \zeta_i,\zeta_i^*<=0
    \end{array} \right)
\end{align*}
$$














To evaluate the performance of the model for various hyperparameter settings a
suitable loss function needs to be defined. An often used cost function $$L(f,x,y)$$ for
regression problems is the Mean Squared Error (MSE):

$$
L(f,x,y) = MSE = \frac{1}{n}\sum_{i=1}^n(f(x)-y)^2
$$

where $$f = A(D)$$ represents the function/model returned by the algorithm $$A$$ when trained on the trainings data set $$D = z_1,...,z_n$$.
$$y$$ are the observed values of the variable being predicted (In terms of hyperparameter optimisation, the observed values are the calculated
losses (e.g. the Mean Squared Error) and the variable $$x$$ describes the hyperparameters.


The performance of the machine learning estimator depends on the hyperparameters and the dataset used for training and validation. Thats the reason, why
we are usually not just choosing a part of the data set as trainings set and another as test set and calculating the squared error for each observation of the
test data set.

To at least mitigate this effect on the performance assessment and get a more generalised assessment, the statistical procedure k-fold cross-validation (CV)
is used in the following.

Therefor the data set is split in $$K$$ subsets. Afterwards $$k-1$$ subsets are used as training data set, one for validation. After the model was build,
the MSE for the validation data set is calculated. This procedure is repeated until each subset has been used once as a validation data set.

Thus, K-models are formed and evaluated in the course of the cross validation. The cross validation score is usually calculated
as an average value from the individual Mean Square Errors.

$$CV(D, \lambda) = \frac{1}{k}\sum_{k=1}^K \frac{1}{m} \sum_{z_i \in T_k}L(A_(\lambda)(D_k), z_i)$$

For example, if you use the sklearn module sklearn.model_selection.cross_val_score and want to use the MSE as the scoring parameter,
you will notice that only the negated MSE can be selected. This is due to the uniform convention of trying to maximise scores in all
cases. Therefore, cost functions are always negated.

<script src="https://gist.github.com/polzerdo55862/ac8cd911802b574693cbe5aa1247c837.js"></script>




The function of the negative cross validation score thus represents the objective function of the mathematical optimization problem.
The objective is to identify the optimal hyperparameter settings, i.e. the hyperparameter values for which the trained models show
the best performance, i.e. the negative cross validation score is maximal.

**Objective function:** The objective function of a mathematical optimization problems is the real-valued function which should be minimized or maximized.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_function_evaluation.png" style="width:100%">
  <figcaption align = "center">
    <b>Bptimization problem- Image by the author</b>
  </figcaption>
</figure>

Since we have no direct knowledge about the analytical form of f(x) at the first moment, we speak of a so-called black-box function.

A black-box function is a system where the internal workings is unknown.
Systems like transistors, engines and human brains are often described as black-box systems.

In our case, the hyperparameters represent the input parameters of the function, for which only it is not directly known how it influences e.g. the cross validation score.
Each point on the function curve must be calculated more or less elaborately.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_function.png" style="width:100%">
  <figcaption align = "center">
    <b>lack-box function – Image by the author (inspired by [Sic18])r</b>
  </figcaption>
</figure>

### Finding the optimal hyperparameter settings

In order to find the optimal hyperparameter settings, we could theoretically calculate the cross validation score
for each possible hyperparameter combination and finally choose the hyperparameter that shows the best performance in the evaluation.

The following figure shows the procedure for the hyperparameter Space $$Epsilon = 0.1-16$$, where $$C$$ constantly takes the value 7.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/black_box_calculation.png" style="width:100%">
  <figcaption align = "center">
    <b>Slample calculation of the black-box function for different hyperparameter settings – Image by the Author</b>
  </figcaption>
</figure>

The hyperparameter optimization process Grid Search works according to this procedure.
Grid Search thus represents the simplest type of hyperparameter optimization.

We thus build up the function approximately with each calculation step bit by bit.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/Build_2d_evaluation.gif" style="width:100%">
  <figcaption align = "center">
    <b>Build up black-box function step for step – Image by the Author</b>
  </figcaption>
</figure>

How close we actually come to the optimum in the area under consideration thus depends decisively on the step size.
However, if we choose a very small step size, have a large number of hyperparameters,
a large data set and possibly also use an algorithm that functions according to a relatively
computationally intensive principle, the computational effort required for the search for the
optimal hyperparameters increases rapidly.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/2d_evaluation_various_step_sizes.gif" style="width:100%">
  <figcaption align = "center">
    <b>Calculated black-box using various step sizes – Image by the Author</b>
  </figcaption>
</figure>


### Grid Search <a name="grid search"/>

As shown in the following figure, we define a "grid" via the hyperparameter Space. If we consider the kernel to be fixed
for the moment, the following two-dimensional hyperparameter Sapce results.
($$C\_min = 1$$, $$C\_max = 50$$, $$Epsilon\_min = 1$$, $$Epsilon\_max = 30$$, $$step\_size = 1$$)

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/grid_search_example.png" style="width:100%">
  <figcaption align = "center">
    <b>Grid search illustration – Image by the Author</b>
  </figcaption>
</figure>

## From Grid to Baysian Optimization <a name="grid to baysian"/>

Definitely a valid approach, at least for so called “cheap” black-box function, where the computation effort to calculate the CV values is low.
But what if the evaluation of the function is costly, so the computational time and/or cost to calculate CV is high?
In this case it may makes sense to think about more “intelligent” ways to find the optimal value. [Cas13]

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/cheap_and_costly_black_box_function.png" style="width:100%">
  <figcaption align = "center">
    <b>Cheap vs. costly black-box functions – Image by the Author</b>
  </figcaption>
</figure>

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

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/hyperparameter_evaluation_2d_gif_step_size_1.1.png" style="width:100%">
  <figcaption align = "center">
    <b>Grid Search example – Image by the Author</b>
  </figcaption>
</figure>

In total, the time needed to compute the needed sample values and the surrogate function,
should be less time-consuming than calculating each point in the hyperparameter space.

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/evaluation_steps.png" style="width:100%">
  <figcaption align = "center">
    <b>Evaluation steps: Grid Search vs. Baysian Optimization – Image by the Author</b>
  </figcaption>
</figure>

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

<figure class="image">
  <img src="/assets/img/posts/06_Bayesian Optimization/Visualize_baysian_opt.gif" style="width:100%">
  <figcaption align = "center">
    <b>Baysian Optimization – Image by the Author</b>
  </figcaption>
</figure>


## References

[Agn20] Agnihotri, Apoorv; Batra, Nipun.  https://distill.pub/2020/bayesian-optimization/. 2020. <br>
[Cas13] Cassilo, Andrea. A Tutorial on Black–Box Optimization. https://www.lix.polytechnique.fr/~dambrosio/blackbox_material/Cassioli_1.pdf. 2013.<br>
[CST79] U.S. Census Service. https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html <br>
[Fah16] Fahrmeir, L.; Heumann, C.; Künstler, R. Statistik: Der Weg zur Datenanalyse. Springer-Lehrbuch. Springer Spektrum, Berlin and Heidelberg, 8., überarbeitete und ergänzte auflage Auflage, 2016. ISBN 978–3–662 50371–3. doi:10.1007/978–3–662–50372–0
[Hel19] Helmenstine, Anne Marie. Bayes Theorem Definition and Examples. https://www.thoughtco.com/bayes-theorem-4155845. 2019. <br>
[Sci18] Sicotte, Xavier. Cross validation estimator. https://stats.stackexchange.com/questions/365224/cross-validation-and-confidence-interval-of-the-true-error/365231#365231. 2018 <br>
[Was21] University of Washington. https://sites.math.washington.edu/. 2021. <br>

