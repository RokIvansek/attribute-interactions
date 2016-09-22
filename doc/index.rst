Interactions add-on
===================

This is the main page of the interactions add-on for Orange 3.
It includes a short description of the add-on, a reference for the scripting library
and a short tutorial on how to use the scripting library.

Attribute interactions
----------------------
This is an Orange 3 add-on. The main mechanism of the add-on is the module :mod:`interactions`.
The module is a re-implementation of a python 2 library written by A. Jakulin for Orange 2.
Utilizing the concept of Shannon entropy, the library is used to find the most informative attribute pairs in a given dataset
or the pairs of attributes that provide additional information, not present when attributes are considered individualy.
You can read more about attribute interactions `here <http://www.stat.columbia.edu/~jakulin/Int/>`_.


Tutorial
--------

This is a short tutorial on how to use the module :mod:`interactions`.

First we load a dataset. The dataset can be discrete or continuous.
In this case we choose the discrete dataset 'zoo' that comes with the Orange library::

   >>> d = Orange.data.Table("zoo")

You can read more about the 'zoo' dataset `here <https://archive.ics.uci.edu/ml/datasets/Zoo>`_.

Then we create an instance of the :class:`interactions.Interactions` class by passing it the chosen dataset::

   >>> inter = Interactions(d)

When creating the object we can choose 2 optional parameters:

* the Laplace parameter **alpha** and
* the discretization object **disc_method**.

Continuous data will be discretized using the **disc_method** discretization method.
For more information on this two parameters take a look at the section `Reference`_.

To get the interactions between the attributes we use the :func:`interactions.Interactions.interaction_matrix` method::

   >>> inter.interaction_matrix()

The mentioned method stores interactions in an object of the type **Orange.misc.distmatrix.DistMatrix** internaly, the values stored
in this matrix are the total relative information gains of all pairs of attributes.

After the :func:`interactions.Interactions.interaction_matrix` method has been called, we can use the
:func:`interactions.Interactions.get_top_att` method to look at the most interesting interactions between attributes in
our dataset::

   >>> best_total = inter.get_top_att(3)

In the line above the :class:`interaction.Interaction` objects for the top three pairs of attributes are created.
The **Interaction** object contains all of the interaction information for the chosen pair of attributes:
the individual info gains, the combined info gain, the mutual information, ...
Because the :class:`interactions.Interaction` objects print nicely, we can simply use::

   >>> print(best_total[0])
   Interaction beetween attributes legs and milk.
   Relative info gain for attribute legs: 0.556274943686
   Relative info gain for attribute milk: 0.403899212505
   Relative info gain for both attributes together: 0.168466861349
   Total relative info gain from attributes and their combination: 0.791707294843

to look at the interaction information about the best pair of attributes in our dataset.


Reference
---------

.. toctree::
   :maxdepth: 1

   source/reference/interactions












	









Widgets
-------

.. toctree::
   :maxdepth: 1

   widgets/mywidget

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

