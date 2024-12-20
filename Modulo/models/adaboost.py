from utils import utils
from sklearn.ensemble import AdaBoostClassifier
from preprocessdata import preprocesssing as pre


class ABClassifier(AdaBoostClassifier):
    """
    An AdaBoost classifier.

    An AdaBoost classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    escenario : string
        CTU-13 scenario identifier to be used

    k : int
        Number of subsets to be used for cross validation

    test_size : float
        Percentage of data used for testing
    """

    def __init__(self, n_estimators=250, escenario='11', k=5, test_size=0.2):
        super().__init__(n_estimators=n_estimators)
        self.escenario = "./database-preprosesing/smote/" + escenario + "/minmax/" + escenario + ".minmax_smote.pickle"
        self.k = k
        self.test_size = test_size

    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(self, X):
        """
        Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        return super().predict(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
           True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        return super().score(X, y)

    def prepare_data(self, cross_val=True):
        """
        Executes the pre-processing of the data to train the model

        Parameters
        ----------
        cross_val : Boolean
            Represents if the cross validation train will be performed or not

        Returns
        -------
            train : array-like of shape (n_samples, n_components)
                Subset of the data used to train the algorithm

            test : array-like of shape (n_samples, n_components)
                Subset of the data used to test the algorithm
        """
        data = './DETECTION SYSTEM/database/*[0123456789].binetflow'
        # {'standard', 'minmax', 'robust', 'max-abs'}
        scalers = {'minmax'}
        # 'under_sampling', 'over_sampling', 'smote', 'svm-smote' 'adasyn' 'no_balanced'
        samplers = ['no_balanced']
        pre.preprocessing(data, scalers, samplers)
        train_data, train_labels = utils.load_and_divide(self.escenario)
        train = []
        test = []
        if not cross_val:
            X_train, X_test, y_train, y_test = utils.train_test_split(train_data, train_labels,
                                                                      test_size=self.test_size)
            train.append([X_train, y_train])
            test.append([X_test, y_test])
        else:
            train, test = utils.create_k(train_data, train_labels, self.k)
        return train, test
