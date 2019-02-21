{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __getitem__, __repr__

   {% block methods %}
   {% endblock %}
