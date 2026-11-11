"""ReactorGenome calculation engine.

Nothing in this package imports Streamlit. Every function takes plain numbers
or arrays and returns plain numbers, arrays or dicts, so the same code runs in
the app, in the tests and in the validation script.

Internal units: concentration mol/L, volume L, flow L/min, time min,
temperature K inside the balances (°C at the edges), energy J.
A is NaOH, B is ethyl acetate, C is sodium acetate, D is ethanol.
"""
