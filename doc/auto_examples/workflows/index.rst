

.. _sphx_glr_auto_examples_workflows:

Tutorials
---------

The following tutorials contain the workflow and code to simulate and visualize commonly
studied M/EEG signals including event related potentials and low frequency rhythms in the
alpha, beta and gamma bands. The tutorial workflows, and all parameter values used,
are based on the detailed tutorials provided on the `HNN GUI website
<https://hnn.brown.edu/tutorials>`_, and reproduce a subset of examples and figures in the
more elaborated GUI tutorials using the HNN-core API.

We strongly recommend that you
first go through the background information provided on the `HNN website
<https://hnn.brown.edu/overview-uniqueness/>`_ and each of the HNN GUI tutorials,
after which the code and instructions in these HNN-core tutorials will be clearer.
All tutorials build from our prior publications investigating the origin of ERPs and
brain rhythms in the somatosensory system, and parameter and data sets are provided to
help users get started.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to simulate a threshold level tactile evoked response, as detailed in the HNN GUI ERP tutorial, using HNN-core. We recommend you first review the GUI tutorial.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_simulate_evoked_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_workflows_plot_simulate_evoked.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">01. Simulate Event Related Potentials (ERPs)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to simulate alpha and beta frequency activity in the alpha/beta complex of the SI mu-rhythm [1]_, as detailed in the HNN GUI alpha and beta tutorial, using HNN-Core.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_simulate_alpha_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_workflows_plot_simulate_alpha.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">02. Simulate Alpha and Beta Rhythms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to simulate gamma rhythms via the well established pyramidal-interneuron-gamma mechanisms [1]_, as detailed in the HNN GUI gamma tutorial, using HNN-Core.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_simulate_gamma_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_workflows_plot_simulate_gamma.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">03. Simulate Gamma Rhythms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to calculate an inverse solution of the median nerve evoked response potential (ERP) in S1 from the MNE somatosensory dataset, and then simulate a biophysical model network that reproduces the observed dynamics. Note that we do not expound on how we came up with the sequence of evoked drives used in this example, rather, we only demonstrate its implementation. For those who want more background on the HNN model and the process used to articulate the proximal and distal drives needed to simulate evoked responses, see the `HNN ERP tutorial`_. The sequence of evoked drives presented here is not part of a current publication but is motivated by prior studies [1]_, [2]_.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_simulate_somato_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_workflows_plot_simulate_somato.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">04. From MEG sensor-space data to HNN simulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how event related potentials (ERP) are modulated by prestimulus beta events. Specifically, this example reproduces Figure 5 from Law et al. 2021 [1]_. To be consistent with the publication, the default network connectivity is altered. These modifications demonstrate a potential mechanism by which transient beta activity in the neocortex can suppress the perceptibility of sensory input. This suppression depends on the timing of the beta event, and the incoming sensory information.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_simulate_beta_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_workflows_plot_simulate_beta.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">05. Simulate beta modulated ERP</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/workflows/plot_simulate_evoked
   /auto_examples/workflows/plot_simulate_alpha
   /auto_examples/workflows/plot_simulate_gamma
   /auto_examples/workflows/plot_simulate_somato
   /auto_examples/workflows/plot_simulate_beta

