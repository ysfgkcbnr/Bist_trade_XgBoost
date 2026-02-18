import pandas as pd
import numpy as np

def adjust_for_splits(df):
    """
    Otomatik bolunme/temettu duzeltmesi.
    Eger bir gunde %25'ten fazla dusus varsa ve bu bir pazar cokusu degilse (devre kesici vs),
    bunu split olarak varsayip gecmis veriyi duzeltir.
    """
    df = df.copy()
    if 'Close' not in df.columns:
        return df
        
    # Yuzdesel degisim hesapla
    pct_change = df['Close'].pct_change()
    
    # %25'ten buyuk dususleri bul (Split genelde %33, %50, %100 vs olur)
    # Normade taban %10 veya %20'dir. %25 ve uzeri teknik olarak normal piyasa sartlarinda imkansiz.
    threshold = -0.25
    split_indices = df[pct_change < threshold].index
    
    # Her split icin
    for date in split_indices:
        if date not in df.index: continue
        loc = df.index.get_loc(date)
        if loc == 0: continue
            
        new_close = df['Close'].iloc[loc]
        prev_close = df['Close'].iloc[loc-1]
        
        if new_close == 0: continue
        ratio = prev_close / new_close
        
        # Eger ratio mantikli bir araliktaysa (orn. 1.2 - 20 arasi)
        if 1.2 <= ratio <= 20.0:
            # Bu tarihten ONCEKI tum verileri ratio'ya bol (fiyatlari indir)
            # Volume dısında
            cols = ['Open', 'High', 'Low', 'Close']
            mask = df.index < date
            for col in cols:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] / ratio
            
    return df
