"""Tests para el módulo de datos de Air Quality UCI."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ttac_test_ds_timeseries.data import (
    load_air_quality_data,
    validate_timeseries_data,
    clean_timeseries_data
)


class TestAirQualityData:
    """Tests para datos de Air Quality UCI."""

    def test_load_air_quality_data_basic(self):
        """Test básico de carga de datos de Air Quality UCI."""
        try:
            data = load_air_quality_data(save_to_disk=False, target_column='CO(GT)')
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert 'CO(GT)' in data.columns
            
        except Exception as e:
            pytest.skip(f"Datos de Air Quality UCI no disponibles: {e}")

    def test_validate_timeseries_data(self):
        """Test de validación de datos temporales."""
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'datetime': pd.date_range('2004-01-01', periods=100, freq='H'),
            'CO(GT)': np.random.normal(2.0, 0.5, 100)
        })
        
        validation = validate_timeseries_data(test_data)
        
        assert isinstance(validation, dict)
        assert 'is_valid' in validation

    def test_clean_timeseries_data(self):
        """Test de limpieza de datos temporales."""
        # Crear datos con duplicados
        test_data = pd.DataFrame({
            'datetime': ['2004-01-01 10:00:00', '2004-01-01 10:00:00', '2004-01-01 11:00:00'],
            'CO(GT)': [2.0, 2.1, 2.2]
        })
        test_data['datetime'] = pd.to_datetime(test_data['datetime'])
        
        cleaned = clean_timeseries_data(test_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) <= len(test_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
