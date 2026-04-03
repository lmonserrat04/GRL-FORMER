import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
# Diccionario de mapeo SITE_ID -> TR (segundos)
# Datos verificados de los protocolos oficiales de NITRC
TR_BY_SITE = {
    "CALTECH": 2.0,
    "CMU": 2.0,
    "KKI": 2.5,
    "LEUVEN_1": 1.67,
    "LEUVEN_2": 1.67,
    "MAX_MUN": 3.0,
    "NYU": 2.0,
    "OHSU": 2.5,
    "OLIN": 1.5,
    "PITT": 1.5,
    "SBL": 2.2,
    "SDSU": 2.0,
    "STANFORD": 2.0,
    "TRINITY": 2.0,
    "UCLA_1": 3.0,
    "UCLA_2": 3.0,
    "UM_1": 2.0,
    "UM_2": 2.0,
    "USM": 2.0,
    "YALE": 2.0
}

def interpolate_timeseries(
    time_fmri_serie: np.ndarray,
    site: str,
    tr_goal: float,
    min_timesteps: int
) -> np.ndarray | None:
    """Interpola una serie temporal fMRI a un TR objetivo y opcionalmente la recorta.
 
    Args:
        ts: Array de forma ``(n_rois, timesteps)``.
        site: Identificador de sitio (debe existir en :data:`TR_BY_SITE`).
        tr_goal: TR objetivo en segundos al que se interpolará la serie.
        min_timesteps: Sujetos con ``timesteps <= min_timesteps`` son descartados.
       
 
    Returns:
        Array de forma ``(n_rois, timesteps_nuevos)`` o ``(n_rois, crop_size)``
        Retorna ``None`` si el sujeto es descartado por tener
        pocos timesteps.
 
    Raises:
      
        KeyError: Si ``site`` no existe en :data:`TR_BY_SITE`.
    """

    nro_timesteps = time_fmri_serie.shape[1]
    if nro_timesteps < min_timesteps:
        return None


    if TR_BY_SITE[site] != tr_goal:
    
        tr_original = TR_BY_SITE[site]

        

        duracion_total = (nro_timesteps - 1) * tr_original

        x = np.linspace(0, duracion_total, nro_timesteps)
        
        nro_pasos_nuevos = int(duracion_total / 2.0) + 1

        x_new = np.arange(0, nro_pasos_nuevos) * 2.0

        y = time_fmri_serie

        f = interp1d(x, y, kind='cubic', axis=1, bounds_error= False , fill_value= 'extrapolate')

        y_new = f(x_new)
        
        return y_new
    
    else:
        return time_fmri_serie



def get_window(time_fmri_serie , window_size:int, mode: str):
    """
    Input
        Matriz de la forma (nro_ROIS, Timesteps)
    Output
        Ventana comenzando desde timestep (random, central) de tamaño window_size

    """
    if mode not in ['random', 'central']:
        raise ValueError("Se esperaba modo de uno de estos tipos: 'random', 'central'")
    

    time_fmri_serie = time_fmri_serie.numpy().copy()


    n_timesteps = time_fmri_serie.shape[1]
    
    max_start = n_timesteps - window_size
    
    if max_start < 0:
        raise ValueError("El window_size es mayor que el tiempo total de la serie.")
    

    if mode == 'random':
        offset = np.random.randint(0, max_start + 1)
    elif mode == 'central':
        offset = max_start // 2
    
   
    window = time_fmri_serie[:, offset : offset + window_size]

    return window


