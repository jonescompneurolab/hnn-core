Glossary
========

.. currentmodule:: hnn_core

The Glossary provides short definitions of HNN-specific vocabulary and
general computational neuroscience concepts. If you think a term is missing,
please consider creating a new issue or opening a pull request to add it.

.. glossary::
    :sorted:


    drive
        An artificial cell or population of artificial cells that only produces
        spikes according to a given pattern and is used to initiate and sustain
        cell spiking in the network

    gid
        The "global" cell ID. A gid is assigned to each cell in the
        network as well as artificial drive cells. Global refers to
        the fact that the cell ID is unique even if the simulation
        is split across cores.

    distal
        Anatomical location that is further away from the soma.
        Distal drives refer to extrinsic connections to the distal dendrites of
        the cortical column, which often originate from indirect thalamic
        nuclei, as well as other cortical regions.

    proximal
        Anatomical location that is closer to the soma.
        Proximal drives refer to extrinsic connections to the proximal dendrites
        of the cortical column, which often originate from direct thalamic
        nuclei.
