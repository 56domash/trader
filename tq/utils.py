
import numpy as np, pandas as pd
def clip01(x): return np.minimum(1.0, np.maximum(0.0, x))
def z_to_01(z): return clip01((z + 1.0)/2.0)
def safe_div(a,b): return a/(b+1e-9)
def to5m(df):
    agg = df.resample('5min', origin='start_day').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    return agg.dropna(how='any')
def anchored_vwap(df_day):
    tp = (df_day['high']+df_day['low']+df_day['close'])/3.0
    cumv = df_day['volume'].replace(0,np.nan).cumsum()
    return (tp*df_day['volume']).cumsum()/(cumv.replace(0,np.nan))
def robust_z(series, clip=3.0):
    s = pd.Series(series, dtype='float64')
    med = np.nanmedian(s); mad = np.nanmedian(np.abs(s-med))
    if mad==0 or np.isnan(mad): z = s-med
    else: z = (s-med)/(1.4826*mad)
    return np.clip(z, -clip, clip)/clip
