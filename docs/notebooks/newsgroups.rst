
20 newsgroups text classification
=================================

In this example we consider a multiclass text classification problem
based on the `20 newsgroups
dataset <http://qwone.com/~jason/20Newsgroups/>`__ which contains the
text of nearly 20,000 newsgroup posts partitioned across 20 different
newsgroups. We fit our classifier by minimizing `multiclass hinge
loss <http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf>`__
combined with `elastic
net <https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf>`__
regularization which combines the :math:`\ell_1` and :math:`\ell_2`
penalty.

.. code:: python

    import numpy as np
    import cvxpy as cp
    import epopt as ep

This dataset is readily available from ``sklearn`` Python package, we
load the training and test data using the 60/40% "by date" split which
makes our results comparable to existing published work.

.. code:: python

    from sklearn.datasets import fetch_20newsgroups
    
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")

Features
--------

The newsgroups data is simply the raw text:

.. code:: python

    print newsgroups_train.data[0]


.. parsed-literal::

    From: lerxst@wam.umd.edu (where's my thing)
    Subject: WHAT car is this!?
    Nntp-Posting-Host: rac3.wam.umd.edu
    Organization: University of Maryland, College Park
    Lines: 15
    
     I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
    
    Thanks,
    - IL
       ---- brought to you by your neighborhood Lerxst ----
    
    
    
    
    


Thus, the first step is to convert this to a set of numerical features
:math:`x_1,\ldots,x_m \in \mathbb{R}^n` that we can use for
classification. We simply employ the standard tf-idf weighting scheme
which weights terms by their term frequency times their inverse document
frequency.

.. code:: python

    from sklearn.feature_extraction import text
    
    vectorizer = text.TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(newsgroups_train.data)
    y = newsgroups_train.target
    Xtest = vectorizer.transform(newsgroups_test.data)
    ytest = newsgroups_test.target

For the purposes of this example, we restrict ourselves to the top 5000
terms which gives a training set of size
:math:`X \in \mathbb{R}^{11314 \times 5000}`:

.. code:: python

    print X.shape


.. parsed-literal::

    (11314, 5000)


Multiclass hinge loss with elastic net regularization
-----------------------------------------------------

Now, we fit the classifier by minimizing multiclass hinge loss combined
with elastic net regularization. Let :math:`\theta_j` for
:math:`j = 1,\ldots,20` denote the weights for class :math:`j`, we fit
the model by solving the optimization problem

.. math::


   \DeclareMathOperator{\minimize}{minimize} \minimize \;\; \sum_{i=1}^m \left( \max_j \; \{\theta_j^Tx_i + 1 - \delta_{j,y_i} \} - \theta_{y_i}^Tx_i \right) + \sum_{j=1}^k \lambda_1 \|\theta_j\|_1 +  \sum_{j=1}^k \lambda_2 \|\theta_j\|_2^2

where :math:`\lambda_1 \ge 0` and :math:`\lambda_2 \ge 0` are
regularization parameters. The :math:`\ell_1` and :math:`\ell_2` penalty
are straightforward to expression in CVXPY, and for multiclass hinge
loss it is most efficient if we write the expression in matrix form

.. code:: python

    def multiclass_hinge_loss(Theta, X, y):
        k = Theta.size[1]
        Y = one_hot(y, k)
        return (cp.sum_entries(cp.max_entries(X*Theta + 1 - Y, axis=1)) -
                cp.sum_entries(cp.mul_elemwise(X.T.dot(Y), Theta)))

For convenience, this definition is provided as part of epsilon, see
```functions.py`` <github.com>`__ for details.

.. code:: python

    # Parameters
    m, n = X.shape
    k = 20
    Theta = cp.Variable(n, k)
    lam1 = 0.1
    lam2 = 1
    
    f = ep.multiclass_hinge_loss(Theta, X, y) + lam1*cp.norm1(Theta) + lam2*cp.sum_squares(Theta)
    prob = cp.Problem(cp.Minimize(f)) 
    ep.solve(prob, verbose=True)
    
    Theta0 = np.array(Theta.value)
    print "Train accuracy:", accuracy(np.argmax(X.dot(Theta0), axis=1), y)
    print "Test accuracy:", accuracy(np.argmax(Xtest.dot(Theta0), axis=1), ytest)


.. parsed-literal::

    Epsilon 0.2.4
    Compiled prox-affine form:
    objective:
      add(
        affine(dense(A)*var(x)),
        non_negative(var(y)),
        affine(kron(dense(B), dense(C))*diag(D)*var(Z)),
        norm_1(var(W)),
        sum_square(var(V)))
    
    constraints:
      zero(add(add(kron(transpose(dense(B)), scalar(1.00))*var(x), scalar(-1.00)*add(kron(scalar(1.00), sparse(K))*var(V), dense(e)*1.00, scalar(-1.00)*const(F))), scalar(-1.00)*var(y)))
      zero(add(var(Z), scalar(-1.00)*var(V)))
      zero(add(var(W), scalar(-1.00)*var(V)))
    Epsilon compile time: 0.0648 seconds
    
    iter=0 residuals primal=8.61e+02 [8.71e+00] dual=8.46e+01 [8.76e+00]
    iter=40 residuals primal=1.18e+00 [4.95e+00] dual=6.83e+00 [8.88e+00]
    Epsilon solve time: 62.8336 seconds
    Train accuracy: 0.970567438572
    Test accuracy: 0.796601168348


Thus, with this straightforward approach feature generation, and simple
bag-of-words model we achieve ~80% accuracy. Note that its well-known
that for this dataset the by date split tends to result in poorer than
expected generalization error (presumably, due to the fact that the
content of a particular newsgroup drifts over time).

Nonetheless, we could no doubt improve upon this result by (for example)
including `higher order
n-grams <http://papers.nips.cc/paper/4932-compressive-feature-learning.pdf>`__,
considering `more sophisticated NLP
features <http://nlp.stanford.edu/wiki/Software/Classifier/20_Newsgroups>`__
and various other approaches to feature engineering...
