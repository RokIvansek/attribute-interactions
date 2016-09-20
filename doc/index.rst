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

The information gain of all individual attributes is computed at the initialization of the object,
it is stored in the dictionary variable called **info_gains**. We can easily take a look at the
information gain of the first five attributes in our dataset::

   >>> for i in range(5):
   >>>    key = inter.data.domain.attributes[i].name
   >>>    print(key, ":", inter.info_gains[key])

The first few rows of the output look like this::

   >>>
   hair : 0.784187779941
   feathers : 0.709519500315
   eggs : 0.824152632249
   milk : 0.965778845947
   airborne : 0.46439873855


To see the actual percentage of the class variable each attribute explains, we need to divide the info gains
with the class entropy. The class entropy was also computed at the initialisation and is stored in the variable
**class_entropy**::

   >>> for i in range(5):
   >>>    key = inter.data.domain.attributes[i].name
   >>>    print(key, ":", inter.info_gains[key]/inter.class_entropy)

Now the values make more sense, since they are relative to the entropy of the class::

   >>>
   hair : 0.327955854597
   feathers : 0.296728768327
   eggs : 0.344669590296
   milk : 0.403899212505
   airborne : 0.194216601012

To get the interactions between attributes we use the :func:`interactions.Interactions.interaction_matrix` method::

   >>> interacts_M = inter.interaction_matrix()

The mentioned method returns an object of the type **Orange.misc.distmatrix.DistMatrix**, the values stored
in this matrix are the relative information gains of all pairs of attributes. Besides this output, the method also stores
all the interactions in a list of :class:`interactions.Interaction` objects called **all_pairs**. These objects contain all
of the interaction information of the chosen two attributes: the individual info gains, the combined info gain,
the mutual information, ...

After the :func:`interactions.Interactions.interaction_matrix` method has been called, we can use the
:func:`interactions.Interactions.get_top_att` method to look at the most interesting interactions between attributes in
our dataset::

   >>> best_total = inter.get_top_att(3, criteria="total")

In the line above we store a subset of **all_pairs** in a new variable **best_total**. This are the top three pairs of
attributes that have the highest information gain. Because the :class:`interactions.Interaction` objects print nicely.
We can simply use::

   >>> print(best_total[0])
   Interaction beetween attributes legs and milk.
   Relative info gain for attribute legs: 0.556274943686
   Relative info gain for attribute milk: 0.403899212505
   Relative info gain for both attributes together: 0.168466861349
   Total relative info gain from attributes and their combination: 0.791707294843

to look at the interaction information about the best pair of attributes in our dataset.

Another kind of attribute pairs that are interesting, are the ones that provide additional gain, that is
not present when the attributes are looked at separately. This are the attributes that have a negative mutual information.
We can find these attributes with the same method as above, just that this time, we specifiy a different criteria::

   >>> best_interaction = inter.get_top_att(3, criteria="interaction")
   >>> print(best_interaction[0])
   Interaction beetween attributes catsize and predator:
   Relative info gain for attribute catsize: 0.127748953466
   Relative info gain for attribute predator: 0.0386518299126
   Relative info gain for both attributes together: -0.0329988599794
   Total relative info gain from attributes and their combination: 0.199399643358


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

