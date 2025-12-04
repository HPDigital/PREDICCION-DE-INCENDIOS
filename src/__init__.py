"""
__init__.py
===========
Módulo principal del proyecto de Predicción de Incendios Forestales

Autor: Herwig Luk Poleyn Paz
Fecha: Diciembre 2023 - Actualizado 2025
"""

__version__ = '1.0.0'
__author__ = 'Herwig Luk Poleyn Paz'
__email__ = 'herwig.poleyn@ucb.edu.bo'
__institution__ = 'Universidad Católica Boliviana - Cochabamba'

from . import data_collection
from . import data_preprocessing

__all__ = [
    'data_collection',
    'data_preprocessing',
]
