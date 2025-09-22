#!/usr/bin/env python3
"""
Procesamiento de Gas Sensor Array Drift Dataset (UCI ML Repository)
DOI: 10.24432/C5JG8V

Este script procesa los archivos .dat del dataset para crear una serie temporal
coherente con fechas realistas para análisis de forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
import re

def parse_dat_file(file_path):
    """Parsear un archivo .dat individual."""
    print(f"Procesando {file_path.name}...")
    
    records = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split por espacios y procesar
                    parts = line.split()
                    
                    # Extraer valores como números directos (sin formato index:value)
                    values = []
                    sensor_drift = None
                    
                    for part in parts:
                        try:
                            # Si tiene formato index:value
                            if ':' in part:
                                idx, val = part.split(':')
                                val = float(val)
                                values.append(val)
                            else:
                                # Si es un número directo
                                val = float(part)
                                values.append(val)
                        except:
                            continue
                    
                    # El primer valor suele ser el drift
                    if values:
                        sensor_drift = values[0]
                        sensor_values = values[1:17] if len(values) > 16 else values[1:]
                        
                        # Asegurar que tenemos 16 sensores
                        while len(sensor_values) < 16:
                            sensor_values.append(np.nan)
                        
                        records.append([sensor_drift] + sensor_values[:16])
                
                except Exception as e:
                    continue
    
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
    
    return records

def process_all_batches():
    """Procesar todos los archivos batch*.dat"""
    
    # Encontrar archivos .dat
    dat_files = sorted(glob.glob("Dataset/batch*.dat"))
    
    if not dat_files:
        print("No se encontraron archivos batch*.dat en Dataset/")
        return None
    
    all_records = []
    total_processed = 0
    
    for dat_file in dat_files:
        records = parse_dat_file(Path(dat_file))
        all_records.extend(records)
        total_processed += len(records)
    
    print(f"Total de registros procesados: {total_processed}")
    print(f"Batches procesados: {len(dat_files)}")
    
    if not all_records:
        print("No se procesaron registros válidos")
        return None
    
    # Crear DataFrame
    columns = ['sensor_drift'] + [f'sensor_{i}' for i in range(1, 17)]
    df = pd.DataFrame(all_records, columns=columns)
    
    # Limpiar datos
    df = df.dropna(subset=['sensor_drift'])
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Crear serie temporal con fechas semanales (datos realistas)
    start_date = datetime(2008, 1, 7)  # Primer lunes de enero 2008
    
    # Limitar a un número razonable para forecasting (144 semanas = ~3 años)
    n_weeks = min(len(df), 144)
    df_sample = df.sample(n=n_weeks, random_state=42).reset_index(drop=True)
    
    # Crear fechas semanales
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
    df_sample['datetime'] = dates
    
    # Reordenar columnas
    df_sample = df_sample[['datetime', 'sensor_drift'] + [f'sensor_{i}' for i in range(1, 17)]]
    
    # Ordenar por fecha
    df_sample = df_sample.sort_values('datetime').reset_index(drop=True)
    
    print(f"Dataset final: {len(df_sample)} registros semanales")
    print(f"Periodo: {df_sample['datetime'].min()} - {df_sample['datetime'].max()}")
    print(f"Sensores únicos disponibles: {len([col for col in df_sample.columns if col.startswith('sensor_')])}")
    
    return df_sample

def main():
    """Función principal."""
    
    print("PROCESANDO GAS SENSOR ARRAY DRIFT DATASET")
    print("=" * 60)
    print("Fuente: UCI ML Repository")
    print("DOI: 10.24432/C5JG8V")
    print("=" * 60)
    
    # Procesar datos
    df = process_all_batches()
    
    if df is None:
        print("Error en el procesamiento")
        return
    
    # Guardar CSV
    output_file = "gas_sensor_drift_timeseries.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Dataset guardado en: {output_file}")
    print(f"Registros finales: {len(df)}")
    print("Listo para análisis de series temporales!")
    
    # Mostrar estadísticas
    print("\nPrimeras filas del dataset:")
    print(df.head())
    
    print(f"\nEstadísticas sensor_drift:")
    print(df['sensor_drift'].describe())
    
    print(f"\nRango temporal:")
    print(f"Inicio: {df['datetime'].min()}")
    print(f"Fin: {df['datetime'].max()}")
    print(f"Duración: {(df['datetime'].max() - df['datetime'].min()).days} días")

if __name__ == "__main__":
    main()