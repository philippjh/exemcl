==========
Python API
==========

.. py:module:: exemcl
.. autoclass:: ExemplarClustering

    :members: models

    .. automethod:: __init__

        Initializes the submodular function of Exemplar-based clustering with the following parameters:

        :param int ground_set: The ground set for the function (usually denoted as :math:`V`).
        :param int precision: Required floating point precision (possible values: ``fp16``, ``fp32`` or ``fp64``).
        :param int device: Computing device to use for function evaluation (possible values: ``gpu`` or ``cpu``). Please keep in mind, that FP16 precision is not available with CPUs.
        :param int worker_count: Number of parallel workers to consider (-1 defaults to all available cores).

    .. method:: __call__(S)

        Evaluates the function value for a single set :math:`S`.

        :param ndarray S:  Input data set :math:`S` represented as data matrix with shape ``[n, d]``.
        :return: Function value :math:`f(S)`.

    .. method:: __call__(S_multi)

        Evaluates the marginal function values for a set of sets :math:`\left\lbrace S_1, \dots, S_n \right\rbrace`.

        :param List[ndarray] S_multi:  Input data sets :math:`\left\lbrace S_1, \dots, S_n \right\rbrace` represented as data matrices with shape ``[n_i, d]`` for each :math:`S_i`.
        :return: Function values :math:`\left\lbrace f(S_1), \dots, f(S_n) \right\rbrace`.

    .. method:: __call__(S, e)

        Evaluates the marginal function value for a single set :math:`S` and a marginal element :math:`e`.

        :param ndarray S:  Input data set :math:`S` represented as data matrix with shape ``[n, d]``.
        :param ndarray e:  Input data vector :math:`e` with shape ``[d, 1]``.
        :return: Marginal function value :math:`f(S \mid e)`.

    .. method:: __call__(S, e_multi)

        Evaluates the marginal function value for a single set :math:`S` and a set of marginal elements :math:`\left\lbrace e_1, \dots, e_n \right\rbrace`.

        :param ndarray S:  Input data set :math:`S` represented as data matrix with shape ``[n, d]``.
        :param List[ndarray] e_multi:  Input data vectors :math:`\left\lbrace e_1, \dots, e_n \right\rbrace` with shape ``[d, 1]`` each.
        :return: Marginal function values :math:`\left\lbrace f(S \mid e_1), \dots, f(S \mid e_n) \right\rbrace`.

    .. method:: __call__(S_multi, e)

        Evaluates the marginal function values for a set of sets :math:`\left\lbrace S_1, \dots, S_n \right\rbrace` and a marginal element :math:`e`.

        :param List[ndarray] S_multi:  Input data sets :math:`\left\lbrace S_1, \dots, S_n \right\rbrace` represented as data matrices with shape ``[n_i, d]`` for each :math:`S_i`.
        :param ndarray e:  Input data vector :math:`e` with shape ``[d, 1]``.
        :return: Marginal function values :math:`\left\lbrace f(S_1 \mid e), \dots, f(S_n \mid e) \right\rbrace`.