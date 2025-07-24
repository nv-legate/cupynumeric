{% for section in sections %}

{{ section.title }}
{{ "~" * section.title|length }}

.. currentmodule:: cupynumeric

.. autosummary::
    :toctree: generated/

.. csv-table::
    :header: NumPy, cupynumeric, single-GPU/CPU, multi-GPU/CPU

    {% for item in section.items -%}
    {{ item.np_ref }}, {{ item.lg_ref }}, {{ item.single }}, {{ item.multi }}
    {% endfor %}

.. rubric:: Summary

Number of NumPy functions: {{ section.np_count }}

Number of functions covered by cupynumeric: {{ section.lg_count }}

{% endfor %}
