{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :inherited-members:
   :special-members: __getitem__, __repr__

   {% block methods %}
   {% endblock %}
