..
  SPDX-FileCopyrightText: 2019-2020 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _config:

##########################################
Configuration
##########################################

PyPSA-ZA has several configuration options which are documented in this section and are collected in a ``config/config.yaml`` file located in the root directory. 
Users should copy the provided default configuration (``config.default.yaml``) and amend their own modifications and assumptions in the user-specific 
configuration file (``config/config.yaml``); confer installation instructions at :ref:`defaultconfig`.

.. _toplevel_cf:

Top-level configuration
=======================

PyPSA-ZA imports the configuration options originally developed in `PyPSA-Eur <https://pypsa-eur.readthedocs.io/en/latest/index.html>`_ and here reported and adapted.

The options here described are collected in a ``config/config.yaml`` file located in the root directory.
Users should copy the provided default configuration (``config.default.yaml``) and amend 
their own modifications and assumptions in the user-specific configuration file (``config/config.yaml``); 
confer installation instructions at :ref:`defaultconfig`.

.. note::
  Credits to PyPSA-Eur and PyPSA-meets-Earth developers for the initial drafting of the configuration documentation here reported

.. _toplevel_cf:

Top-level configuration
=======================

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :lines: 1-5,25-31

..
   .. csv-table::
      :header-rows: 1
      :widths: 25,7,22,30
      :file: configtables/toplevel.csv

.. _scenario:

``scenario``
============

It is common conduct to analysis of energy system optimisation models for **multiple scenarios** for a variety of reasons,
e.g. assessing their sensitivity towards changing the temporal and/or geographical resolution or investigating how
investment changes as more ambitious greenhouse-gas emission reduction targets are applied.

The ``scenario`` section is an extraordinary section of the config file
that is strongly connected to the :ref:`wildcards` and is designed to
facilitate running multiple scenarios through a single command

.. code:: bash

    snakemake -j 1 solve_all_networks

For each wildcard, a **list of values** is provided. The rule ``solve_all_networks`` will trigger the rules for creating ``results/networks/solved_{model_file}_{regions}_{resarea}_l{ll}_{opts}.nc`` for **all combinations** of the provided wildcard values as defined by Python's `itertools.product(...) <https://docs.python.org/2/library/itertools.html#itertools.product>`_ function that snakemake's `expand(...) function <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#targets>`_ uses.

An exemplary dependency graph (starting from the simplification rules) then looks like this:

.. image:: img/scenarios.png

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: scenario:
   :end-before: data:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/scenario.csv

.. _snapshots_cf:

``snapshots``- now specified in config/model_file.xlsx
===============================================

Specifies the temporal range to build an energy system model for as arguments to `pandas.date_range <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_

.. .. literalinclude:: ../config.default.yaml
     :language: yaml
     :start-at: years:
     :end-before: electricity:

.. .. csv-table::
     :header-rows: 1
     :widths: 25,7,22,30
     :file: configtables/snapshots.csv

.. _electricity_cf:

``electricity``
===============

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: electricity:
   :end-before: respotentials:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/electricity.csv

.. warning::
    Carriers in ``conventional_carriers`` must not also be in ``extendable_carriers``.

.. _atlite_cf:

``atlite``
==========

Define and specify the ``atlite.Cutout`` used for calculating renewable potentials and time-series. All options except for ``features`` are directly used as `cutout parameters <https://atlite.readthedocs.io/en/latest/ref_api.html#cutout>`_.

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: atlite:
   :end-before: renewable:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/atlite.csv

.. _renewable_cf:

``renewable``
=============

``onwind``
----------

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: renewable:
   :end-before: solar:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/onwind.csv

``solar``
---------------

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at:   solar:
   :end-before:   hydro_inflow:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solar.csv

``hydro``
---------------

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at:   hydro_inflow:
   :end-before: lines:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/hydro.csv

.. _lines_cf:

``lines``
=============

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: lines:
   :end-before: links:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/lines.csv

.. _links_cf:

``links``
=============

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: links:
   :end-before: augmented_line_connection:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/links.csv

.. _costs_cf:

``costs``
=============

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: costs:
   :end-before: tsam_clustering:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/costs.csv

.. note::
    To change cost assumptions in more detail (i.e. other than ``marginal_cost`` and ``capital_cost``), consider modifying cost assumptions directly in ``data/bundle/costs.csv`` as this is not yet supported through the config file.
    You can also build multiple different cost databases. Make a renamed copy of ``data/bundle/costs.csv`` (e.g. ``data/bundle/costs-optimistic.csv``) and set the variable ``COSTS=data/bundle/costs-optimistic.csv`` in the ``Snakefile``.

.. _solving_cf:

``solving``
=============

``options``
-----------

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: solving:
   :end-before:   solver:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solving-options.csv

``solver``
----------

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at:   solver:
   :end-before: plotting:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/solving-solver.csv

.. _plotting_cf:

``plotting``
=============

.. literalinclude:: ../config.default.yaml
   :language: yaml
   :start-at: plotting:

.. csv-table::
   :header-rows: 1
   :widths: 25,7,22,30
   :file: configtables/plotting.csv