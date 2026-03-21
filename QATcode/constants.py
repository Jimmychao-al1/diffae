"""
Constants and policies shared across QAT pipeline.
Public API of other modules must remain unchanged; these are internal aids.
"""

# First/last 8-bit policy used by Step4/5: set first 3 quant modules and the last 1 module to 8-bit
FIRST_LAST_8BIT_POLICY = {
    "name": "first3_last1",
    "first_count": 3,
    "last_count": 1,
}


