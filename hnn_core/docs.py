"""The documentation functions."""

import sys

docdict = dict()
# Define docdicts

docdict[
    "net"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "fname"
] = """
fname : str | Path object
    Full path to the output file (.hdf5)
"""

docdict[
    "overwrite"
] = """
overwrite : Boolean
    True : Overwrite existing file
    False : Throw error if file already exists
"""

docdict[
    "save_unsimulated"
] = """
save_unsimulated : Boolean
    True : Do not save the Network simulation output
    False : Save complete Network as provided in input
"""

docdict[
    "read_raw"
] = """
read_raw : Boolean
    True : Read unsimulated network
    False : Read simulated network
"""

docdict[
    "cell_description"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "gid_range_description"
] = """
start : int
    Start of the gid_range
stop : int
    End of the gid_range
"""

docdict[
    "external_drive_description"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "external_bias_description"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "connection_description"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "extracellular_array_description"
] = """
net : Instance of Network object
    The Network object
"""

docdict[
    "network_file_content_description"
] = """
object_type : str
    Type of object (Network) saved
N_pyr_x : int
N_pyr_y : int
threshold : float
    Firing threshold of all cells.
celsius : float
cell_types : dict of Cell Object
    key : name of cell type
    value : Cell object
gid_ranges : dict of dict
    key : cell name or drive name
    value : dict
pos_dict : dict
    key : cell type name
    value : All co-ordintes of the cell types
cell_response : Instance of Cell Response Object
    The Cell Response object
external_drives : dict of dict
    key : external drive name
    value : dict
external_biases : dict of dict
    key : external bias name
    value : dict
connectivity : list of dict
rec_arrays : dict of Extracellular Arrays
    key : extracellular array name
    value : Instance of Extracellular Array object
delay : float
    Synaptic delay in ms.
"""

docdict_indented = {}


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError("Error documenting %s:\n%s" % (funcname, str(exp)))
    return f


##############################################################################
# Utilities for docstring manipulation.


def copy_doc(source):
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B (A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError("Cannot copy docstring: docstring was empty.")
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper


def copy_function_doc_to_method_doc(source):
    """Use the docstring from a function as docstring for a method.

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator. Additionally, the first parameter
    specified in the docstring of the source function is removed in the new
    docstring.

    This decorator is useful when implementing a method that just calls a
    function.  This pattern is prevalent in for example the plotting functions
    of MNE.

    Parameters
    ----------
    source : function
        Function to copy the docstring from.

    Returns
    -------
    wrapper : function
        The decorated method.

    Notes
    -----
    The parsing performed is very basic and will break easily on docstrings
    that are not formatted exactly according to the ``numpydoc`` standard.
    Always inspect the resulting docstring when using this decorator.

    Examples
    --------
    >>> def plot_function(object, a, b):
    ...     '''Docstring for plotting function.
    ...
    ...     Parameters
    ...     ----------
    ...     object : instance of object
    ...         The object to plot
    ...     a : int
    ...         Some parameter
    ...     b : int
    ...         Some parameter
    ...     '''
    ...     pass
    ...
    >>> class A:
    ...     @copy_function_doc_to_method_doc(plot_function)
    ...     def plot(self, a, b):
    ...         '''
    ...         Notes
    ...         -----
    ...         .. versionadded:: 0.13.0
    ...         '''
    ...         plot_function(self, a, b)
    >>> print(A.plot.__doc__)
    Docstring for plotting function.
    <BLANKLINE>
        Parameters
        ----------
        a : int
            Some parameter
        b : int
            Some parameter
    <BLANKLINE>
            Notes
            -----
            .. versionadded:: 0.13.0
    <BLANKLINE>
    """

    def wrapper(func):
        doc = source.__doc__.split("\n")
        if len(doc) == 1:
            doc = doc[0]
            if func.__doc__ is not None:
                doc += func.__doc__
            func.__doc__ = doc
            return func

        # Find parameter block
        for line, text in enumerate(doc[:-2]):
            if (text.strip() == "Parameters" and
               doc[line + 1].strip() == "----------"):
                parameter_block = line
                break
        else:
            # No parameter block found
            raise ValueError(
                "Cannot copy function docstring: no parameter "
                "block found. To simply copy the docstring, use "
                "the @copy_doc decorator instead."
            )

        # Find first parameter
        for line, text in enumerate(doc[parameter_block:], parameter_block):
            if ":" in text:
                first_parameter = line
                parameter_indentation = len(text) - len(text.lstrip(" "))
                break
        else:
            raise ValueError(
                "Cannot copy function docstring: no parameters "
                "found. To simply copy the docstring, use the "
                "@copy_doc decorator instead."
            )

        # Find end of first parameter
        for line, text in enumerate(doc[first_parameter + 1:],
                                    first_parameter + 1):
            # Ignore empty lines
            if len(text.strip()) == 0:
                continue

            line_indentation = len(text) - len(text.lstrip(" "))
            if line_indentation <= parameter_indentation:
                # Reach end of first parameter
                first_parameter_end = line

                # Of only one parameter is defined, remove the Parameters
                # heading as well
                if ":" not in text:
                    first_parameter = parameter_block

                break
        else:
            # End of docstring reached
            first_parameter_end = line
            first_parameter = parameter_block

        # Copy the docstring, but remove the first parameter
        doc = (
            "\n".join(doc[:first_parameter]) +
            "\n" +
            "\n".join(doc[first_parameter_end:])
        )
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func

    return wrapper


def _indentcount_lines(lines):
    """Compute minimum indent for all lines in line list."""
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno
