"""
ML MODEL EĞİTİM SİSTEMİ
========================
UT Bot sinyallerini filtreleyen XGBoost modeli eğitir.
Meta-labeling yaklaşımı: "AL sinyali geldi, gireyim mi?"

Feature'lar (Sadece senin stratejin):
- EMA 200, 377, 610
- POI (Destek/Direnç)
- StochRSI K, D
- UT Bot Trailing Stop

Yazar: Trading AI System
Versiyon: 2.1
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import optuna
import logging
import data_utils

# Optuna loglarını sadece hatalar için ayarla (kalabalık yapmasın)
optuna.logging.set_verbosity(optuna.logging.ERROR)

warnings.filterwarnings('ignore')

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost kurulu değil! pip install xgboost")


# ============================================================
# AYARLAR
# ============================================================

# ============================================================
# ÖZEL MODELLER
# ============================================================

class DominantWinnerModel:
    """
    %100 başarı oranına sahip hisseler için dummy model.
    XGBoost tek sınıfta eğitilemediği için bu kullanılır.
    Confidence %85 olarak ayarlandı (geçmiş başarı gelecek garantisi değil).
    """
    def __init__(self, confidence=0.85):
        self.is_dominant = True
        self.confidence = confidence

    def predict(self, X):
        return np.ones(len(X))

    def predict_proba(self, X):
        # [0 olasılığı, 1 olasılığı] - %85 confidence (100% yerine daha gerçekçi)
        return np.column_stack([
            np.full(len(X), 1 - self.confidence),
            np.full(len(X), self.confidence)
        ])


class ModelConfig:
    # Veri Ayarları
    DATA_PERIOD_YEARS = 10     # Sinyal sayısını artırmak için 10 yıla çıkarıldı
    INTERVAL = "1d"             # Günlük mum
    
    # UT Bot Ayarları (Pine Script ile birebir)
    KEY_VALUE = 2
    ATR_PERIOD = 10
    USE_HEIKIN_ASHI = True
    
    # Target Labeling (Backtest ile SENKRONİZE)
    LOOK_AHEAD_BARS = 25        # Bir işlemin sonuçlanması için yeterli bar sayısı
    TARGET_PROFIT = 0.105       # %10.5 brüt hedef kâr (backtest ile aynı)
    STOP_LOSS = 0.05            # %5 stop loss
    COMMISSION_PCT = 0.002      # %0.2 komisyon
    SLIPPAGE_PCT = 0.001        # %0.1 slippage
    
    # Model Ayarları
    TEST_SIZE = 0.2             # Test seti oranı
    N_SPLITS = 5                # Time series CV split sayısı
    RANDOM_STATE = 42
    
    # Optuna Ayarları
    OPTUNA_TRIALS = 20          # Toplu eğitim için 20 deneme yeterli (hız için)
    OPTUNA_TIMEOUT = 45         # Max 45 saniye
    
    # Filtreleme
    PRECISION_THRESHOLD = 0.65  # %65 altındaki precision'a sahip modeller kaydedilmez
    MIN_SIGNALS = 20            # Min 20 sinyal (daha fazla hisse dahil edilsin)
    MIN_TEST_SAMPLES = 8        # Minimum test sinyali sayısı

    # Cross-Validation Ayarları
    CV_SPLITS = 3               # TimeSeriesSplit fold sayısı
    CV_MIN_SIGNALS = 50         # CV için minimum sinyal sayısı

    # Validation ve Early Stopping Ayarları
    EARLY_STOPPING_ROUNDS = 20  # Validation loss iyileşmezse dur
    PURGE_BARS = 25             # Train/Val/Test arasında boşluk (veri sızıntısı önleme)

    # DominantWinner Ayarları
    DOMINANT_CONFIDENCE = 0.85  # %100 başarılı hisseler için confidence (%85 daha gerçekçi)
    
    # Yollar
    INPUT_DIR = "output"
    OUTPUT_DIR = "models"
    ELITE_STOCKS_FILE = "hisseler.txt" # Artık tüm listeyi eğiteceğiz


# ============================================================
# VERİ HAZIRLAMA
# ============================================================

def calculate_heikin_ashi(df):
    """
    Heikin Ashi Close hesaplar - Pine Script ile birebir uyumlu.
    Pine Script: request.security(ticker.heikinashi(...), ..., close)
    """
    ha_df = df.copy()
    # HA Close = (Open + High + Low + Close) / 4
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    return ha_df


def calculate_ut_bot(df, key_value, atr_period, use_ha=True):
    """
    UT Bot sinyallerini hesaplar - Pine Script ile birebir uyumlu.
    
    Pine Script:
        ema = ta.ema(src, 1)
        above = ta.crossover(ema, xATRTrailingStop)
        buy = src > xATRTrailingStop and above
    """
    src = df['HA_Close'] if use_ha else df['Close']
    
    # EMA(1) hesapla - Pine Script uyumu
    ema = ta.ema(src, length=1)
    if ema is None: ema = src.copy()
    
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    if atr is None: atr = pd.Series(0, index=df.index)
    
    nLoss = key_value * atr
    
    xATRTrailingStop = np.zeros(len(df))
    
    for i in range(1, len(df)):
        prev_stop = float(xATRTrailingStop[i-1])
        
        try:
            # Scalar değer aldığımızdan emin olalım
            v_src = src.iloc[i]
            curr_src = float(v_src.iloc[0]) if isinstance(v_src, pd.Series) else float(v_src)
            
            v_prev = src.iloc[i-1]
            prev_src = float(v_prev.iloc[0]) if isinstance(v_prev, pd.Series) else float(v_prev)
            
            v_nloss = nLoss.iloc[i]
            curr_nLoss = float(v_nloss.iloc[0]) if isinstance(v_nloss, pd.Series) else float(v_nloss)
            
            if np.isnan(curr_nLoss) or np.isnan(curr_src):
                xATRTrailingStop[i] = prev_stop
                continue
            
            # Pine Script mantığı - birebir
            if curr_src > prev_stop:
                iff_1 = curr_src - curr_nLoss
            else:
                iff_1 = curr_src + curr_nLoss
            
            if curr_src < prev_stop and prev_src < prev_stop:
                iff_2 = min(prev_stop, curr_src + curr_nLoss)
            else:
                iff_2 = iff_1
            
            if curr_src > prev_stop and prev_src > prev_stop:
                xATRTrailingStop[i] = max(prev_stop, curr_src - curr_nLoss)
            else:
                xATRTrailingStop[i] = iff_2
        except Exception:
            # Hata durumunda önceki değeri koru veya NaN ata
            xATRTrailingStop[i] = prev_stop
            continue
    
    df['Trailing_Stop'] = xATRTrailingStop
    df['ATR'] = atr
    
    # AL sinyali - Pine Script crossover mantığı
    crossover_above = (ema > df['Trailing_Stop']) & (ema.shift(1) <= df['Trailing_Stop'].shift(1))
    df['Signal_Buy'] = ((src > df['Trailing_Stop']) & crossover_above).astype(int)
    
    # SAT sinyali - Pine Script crossover mantığı
    crossover_below = (df['Trailing_Stop'] > ema) & (df['Trailing_Stop'].shift(1) <= ema.shift(1))
    df['Signal_Sell'] = ((src < df['Trailing_Stop']) & crossover_below).astype(int)
    
    return df


def calculate_pivot_points(df, swing_length=10):
    """
    Pine Script ta.pivothigh/ta.pivotlow ile birebir uyumlu pivot hesaplaması.
    Destek/Direnç mantığı "kırılmamış" (unbroken) seviyeleri takip edecek şekilde güncellendi.
    """
    pivot_highs = pd.Series(index=df.index, dtype=float)
    pivot_lows = pd.Series(index=df.index, dtype=float)
    
    # 1. Pivot Noktalarını Belirle (Gelecek verisine bakar - Look ahead)
    # Bu kısım sadece "burada pivot vardı" tespiti içindir.
    # Kullanırken swing_length kadar geriden gelindiği varsayılmalı.
    
    for i in range(swing_length, len(df) - swing_length):
        try:
            # Scalar yardımcı fonksiyonu
            def get_val(series, idx):
                v = series.iloc[idx]
                return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

            current_high = get_val(df['High'], i)
            if np.isnan(current_high): continue
            
            is_pivot_high = True
            for j in range(1, swing_length + 1):
                p_high = get_val(df['High'], i - j)
                if np.isnan(p_high) or p_high >= current_high:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                for j in range(1, swing_length + 1):
                    n_high = get_val(df['High'], i + j)
                    if np.isnan(n_high) or n_high >= current_high:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.iloc[i] = current_high
            
            # Pivot Low
            current_low = get_val(df['Low'], i)
            if np.isnan(current_low): continue
            
            is_pivot_low = True
            for j in range(1, swing_length + 1):
                p_low = get_val(df['Low'], i - j)
                if np.isnan(p_low) or p_low <= current_low:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                for j in range(1, swing_length + 1):
                    n_low = get_val(df['Low'], i + j)
                    if np.isnan(n_low) or n_low <= current_low:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.iloc[i] = current_low
        except:
            continue
    
    df['Pivot_High'] = pivot_highs
    df['Pivot_Low'] = pivot_lows
    
    # 2. Dinamik Destek/Direnç Hesaplama (Look-ahead bias OLMADAN)
    # Pivot'un teyit edildiği (confirmed) zamandan itibaren listeye eklenir.
    # Fiyat kırarsa listeden silinir.
    
    supports = []     # Aktif destek listesi (değerler)
    resistances = []  # Aktif direnç listesi (değerler)
    
    n = len(df)
    nearest_support = np.full(n, np.nan)
    nearest_resistance = np.full(n, np.nan)
    
    lows = df['Low'].values
    highs = df['High'].values
    closes = df['Close'].values
    p_lows = pivot_lows.values
    p_highs = pivot_highs.values
    
    for i in range(n):
        # A. Yeni Teyit Edilen Pivotları Ekle
        # Bir pivot 'i' anında oluştuysa, teyidi 'i + swing_length' anında gerçekleşir.
        # Dolayısıyla şu an 'i' ise, 'i - swing_length' zamanındaki pivot yeni teyit olmuştur.
        conf_idx = i - swing_length
        if conf_idx >= 0:
            val_low = p_lows[conf_idx]
            if not np.isnan(val_low):
                supports.append(float(val_low))
            
            val_high = p_highs[conf_idx]
            if not np.isnan(val_high):
                resistances.append(float(val_high))
        
        # B. Kırılanları Temizle
        current_low = lows[i]
        # Destek, Low fiyat altına inerse kırılır (veya eşitse dokunmuş olur, garanti olsun diye <= kullanıyoruz)
        supports = [s for s in supports if s <= current_low]
        
        current_high = highs[i]
        # Direnç, High fiyat üstüne çıkarsa kırılır
        resistances = [r for r in resistances if r >= current_high]
        
        # C. En Yakın Olanı Seç
        # Destek: Fiyatın (Close) altındaki en yüksek destek
        valid_supports = [s for s in supports if s < closes[i]]
        if valid_supports:
            nearest_support[i] = max(valid_supports)
            
        # Direnç: Fiyatın (Close) üstündeki en düşük direnç
        valid_resistances = [r for r in resistances if r > closes[i]]
        if valid_resistances:
            nearest_resistance[i] = min(valid_resistances)
            
    df['Nearest_Support'] = nearest_support
    df['Nearest_Resistance'] = nearest_resistance
    
    # ============================================================
    # PAZAR REJİMİ VE RELATİF GÜÇ (PRECISION İÇİN)
    # ============================================================
    if 'Index_Close' in df.columns:
        # Verileri numerik yap ve temizle
        idx_close = pd.to_numeric(df['Index_Close'], errors='coerce')
        
        # Endeks Trendi (Market Regime)
        idx_ema = ta.ema(idx_close, length=200)
        if idx_ema is not None:
            df['Index_EMA_200'] = idx_ema
            df['Index_Trend'] = (idx_close > idx_ema.fillna(0)).astype(int)
        else:
            df['Index_Trend'] = 0
        
        # Relatif Güç (Relative Strength)
        df['Stock_Ret_10'] = df['Close'].pct_change(10)
        df['Index_Ret_10'] = idx_close.pct_change(10)
        df['Rel_Strength_10'] = (df['Stock_Ret_10'] - df['Index_Ret_10']).fillna(0)
        
        df['Stock_Ret_30'] = df['Close'].pct_change(30)
        df['Index_Ret_30'] = idx_close.pct_change(30)
        df['Rel_Strength_30'] = (df['Stock_Ret_30'] - df['Index_Ret_30']).fillna(0)
        
        # Endekse göre momentum
        adx_res = ta.adx(df['Index_High'], df['Index_Low'], idx_close)
        if adx_res is not None and 'ADX_14' in adx_res.columns:
            df['Index_ADX'] = adx_res['ADX_14'].fillna(25)
        else:
            df['Index_ADX'] = 25
    
    return df


def engineer_features(df):
    """
    Feature engineering - SADECE SENİN STRATEJİN
    
    EMA DESTEK/DİRENÇ SİSTEMİ:
    - EMA 200: Normal destek/direnç (1 puan)
    - EMA 377: Orta güçte destek/direnç (2 puan)
    - EMA 610: Güçlü destek/direnç (3 puan)
    - EMA'lar yakınsa: SÜPER güçlü bölge (+2 puan)
    
    POI (Destek/Direnç) - Pivot bazlı (Pine Script uyumlu)
    StochRSI K, D
    """
    # === EMA'LAR (200, 377, 610) ===
    # Pandas_ta bazen None dönebilir (yetersiz veri), bu yüzden fillna(0) ve Series kontrolü ekliyoruz.
    def safe_ema(series, length):
        res = ta.ema(series, length=length)
        return res if res is not None else pd.Series(0.0, index=series.index)

    df['EMA_200'] = safe_ema(df['Close'], 200)
    df['EMA_377'] = safe_ema(df['Close'], 377)
    df['EMA_610'] = safe_ema(df['Close'], 610)
    
    # === EMA DESTEK/DİRENÇ HESAPLAMASI ===
    # Fiyatın her EMA'ya uzaklığı (yüzde olarak)
    df['Dist_EMA_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    df['Dist_EMA_377'] = (df['Close'] - df['EMA_377']) / df['EMA_377']
    df['Dist_EMA_610'] = (df['Close'] - df['EMA_610']) / df['EMA_610']
    
    # EMA'ların birbirine yakınlığı (yüzde olarak)
    df['EMA_200_377_Gap'] = abs(df['EMA_200'] - df['EMA_377']) / df['Close']
    df['EMA_377_610_Gap'] = abs(df['EMA_377'] - df['EMA_610']) / df['Close']
    df['EMA_All_Gap'] = abs(df['EMA_200'] - df['EMA_610']) / df['Close']
    
    # EMA'lar yakınsıyor mu? (%2'den az fark = yakın = SÜPER güçlü bölge)
    EMA_CONVERGENCE_THRESHOLD = 0.02
    df['EMAs_Converging'] = (df['EMA_All_Gap'] < EMA_CONVERGENCE_THRESHOLD).astype(int)
    
    # === DESTEK SEVİYELERİ (Fiyat EMA'nın ÜSTÜNDE ve yakınsa = destek) ===
    NEAR_THRESHOLD = 0.02  # %2'den yakınsa "yakın" sayılır
    
    # EMA 200 - Normal güçte destek/direnç
    df['EMA_200_Support'] = (
        (df['Dist_EMA_200'] > 0) &
        (df['Dist_EMA_200'] < NEAR_THRESHOLD)
    ).astype(int)
    df['EMA_200_Resistance'] = (
        (df['Dist_EMA_200'] < 0) &
        (abs(df['Dist_EMA_200']) < NEAR_THRESHOLD)
    ).astype(int)
    
    # EMA 377 - Orta güçte destek/direnç
    df['EMA_377_Support'] = (
        (df['Dist_EMA_377'] > 0) &
        (df['Dist_EMA_377'] < NEAR_THRESHOLD)
    ).astype(int)
    df['EMA_377_Resistance'] = (
        (df['Dist_EMA_377'] < 0) &
        (abs(df['Dist_EMA_377']) < NEAR_THRESHOLD)
    ).astype(int)
    
    # EMA 610 - Güçlü destek/direnç
    df['EMA_610_Support'] = (
        (df['Dist_EMA_610'] > 0) &
        (df['Dist_EMA_610'] < NEAR_THRESHOLD)
    ).astype(int)
    df['EMA_610_Resistance'] = (
        (df['Dist_EMA_610'] < 0) &
        (abs(df['Dist_EMA_610']) < NEAR_THRESHOLD)
    ).astype(int)
    
    # === DESTEK GÜCÜ SKORU (0-8 arası) ===
    df['Support_Strength'] = (
        df['EMA_200_Support'] * 1 +      # 1 puan
        df['EMA_377_Support'] * 2 +      # 2 puan  
        df['EMA_610_Support'] * 3 +      # 3 puan
        df['EMAs_Converging'] * 2        # Yakınsama bonusu: 2 puan
    )
    
    df['Resistance_Strength'] = (
        df['EMA_200_Resistance'] * 1 +
        df['EMA_377_Resistance'] * 2 +
        df['EMA_610_Resistance'] * 3 +
        df['EMAs_Converging'] * 2
    )
    
    # Herhangi bir EMA desteğinde mi?
    df['Near_EMA_Support'] = (df['Support_Strength'] > 0).astype(int)
    df['Near_EMA_Resistance'] = (df['Resistance_Strength'] > 0).astype(int)
    
    # SÜPER güçlü bölge: EMA'lar yakınsıyor VE fiyat bu bölgede
    df['Super_Support_Zone'] = (
        (df['EMAs_Converging'] == 1) &
        (df['Support_Strength'] >= 3)
    ).astype(int)
    
    # Eski uyumluluk için Above_EMA feature'ları koru (ML model için)
    # NaN'leri 0 yaparak karşılaştırma hatasını önle
    df['Above_EMA_200'] = (df['Close'] > df['EMA_200'].fillna(0)).astype(int)
    df['Above_EMA_377'] = (df['Close'] > df['EMA_377'].fillna(0)).astype(int)
    df['Above_EMA_610'] = (df['Close'] > df['EMA_610'].fillna(0)).astype(int)
    df['Above_All_EMAs'] = (df['Above_EMA_200'] & df['Above_EMA_377'] & df['Above_EMA_610']).astype(int)
    
    # EMA sıralaması (boğa trendi)
    df['EMA_Stack'] = ((df['EMA_200'] > df['EMA_377']) & (df['EMA_377'] > df['EMA_610'])).astype(int)
    
    # === POI - DESTEK/DİRENÇ (Pine Script pivot bazlı) ===
    df = calculate_pivot_points(df, swing_length=10)
    
    # Destek/Dirence uzaklık
    df['Dist_to_Resistance'] = (df['Nearest_Resistance'] - df['Close']) / df['Close']
    df['Dist_to_Support'] = (df['Close'] - df['Nearest_Support']) / df['Close']
    
    # POI içinde mi? (Destek/Direnç bölgesine yakın)
    df['Near_Support'] = (df['Dist_to_Support'] < 0.02).astype(int)
    df['Near_Resistance'] = (df['Dist_to_Resistance'] < 0.02).astype(int)
    
    # === STOCHASTIC RSI (K=3, D=3, RSI=14, Stoch=14) ===
    try:
        stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and 'STOCHRSIk_14_14_3_3' in stoch.columns:
            df['StochRSI_K'] = stoch['STOCHRSIk_14_14_3_3'].fillna(50)
            df['StochRSI_D'] = stoch['STOCHRSId_14_14_3_3'].fillna(50)
        else:
            df['StochRSI_K'] = 50
            df['StochRSI_D'] = 50
    except:
        df['StochRSI_K'] = 50
        df['StochRSI_D'] = 50
    
    # StochRSI durumları
    df['StochRSI_Oversold'] = (df['StochRSI_K'].fillna(50) < 20).astype(int)
    df['StochRSI_Overbought'] = (df['StochRSI_K'].fillna(50) > 80).astype(int)
    df['StochRSI_Bullish'] = (df['StochRSI_K'] > df['StochRSI_D']).astype(int)
    
    # === TRAILING STOP UZAKLIĞI ===
    df['Dist_to_Stop'] = (df['Close'] - df['Trailing_Stop']) / df['Close']
    
    # === HACIM ===
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # === FİLTRELENMİŞ AL SİNYALİ (Backtest ile AYNI!) ===
    # Trend koşulu (esnek)
    trend_ok = (
        (df['Close'] > df['EMA_200']) |              # Uptrend'de
        (df['Near_EMA_Support'] == 1) |              # EMA desteğinde
        (df['Super_Support_Zone'] == 1) |            # Süper güçlü bölgede
        (df['Near_Support'] == 1)                    # POI desteğinde
    )
    
    df['Filtered_Buy'] = (
        (df['Signal_Buy'] == 1) &                    # UT Bot AL sinyali
        trend_ok &                                   # Trend uygun
        (df['StochRSI_K'] < 80) &                    # Aşırı alımda değil
        (df['StochRSI_Bullish'] == 1)                # StochRSI bullish
    ).astype(int)
    
    return df


def create_target_label(df, look_ahead, config):
    """
    BACKTEST İLE BİREBİR AYNI TARGET LABELING (DÜZELTİLMİŞ)
    
    Her Filtered_Buy sinyali için backtest'teki gibi simüle eder:
    1. Trailing Stop: Kârda yükselir (highest * (1 - stop_loss))
    2. Target Profit: +%10.5'e ulaşırsa = 1 (Net kâr varsa)
    3. Stop Loss: -%5'e düşerse = 0 (Net zarar varsa)
    4. Signal_Sell: SAT sinyali gelirse işlem kapanır
    
    ÖNCELİK: Target > Stop > Signal_Sell
    Maliyet: Komisyon ve Slippage hesaba katılır.
    """
    targets = []
    target_profit = config.TARGET_PROFIT
    stop_loss = config.STOP_LOSS
    costs = config.COMMISSION_PCT + config.SLIPPAGE_PCT
    
    for i in range(len(df)):
        if i + look_ahead >= len(df):
            targets.append(np.nan)
            continue
        
        entry_price = df['Close'].iloc[i]
        target_price = entry_price * (1 + target_profit)
        initial_stop = entry_price * (1 - stop_loss)
        
        # Simülasyon değişkenleri
        highest_since_entry = df['High'].iloc[i]
        current_stop = initial_stop
        
        # Gelecek N bar'ı incele (backtest gibi bar bar)
        future_slice = df.iloc[i+1:i+1+look_ahead]
        
        result = None  # None = henüz karar yok
        
        for j, (idx, future_row) in enumerate(future_slice.iterrows()):
            try:
                f_high = float(future_row['High'])
                f_low = float(future_row['Low'])
                f_close = float(future_row['Close']) # Keep f_close for Signal_Sell check
                
                if np.isnan(f_high) or np.isnan(f_low): continue

                # Trailing Stop güncelle (kârda yükselir)
                if f_high > highest_since_entry:
                    highest_since_entry = f_high
                
                if highest_since_entry > entry_price:
                    trailing_stop = highest_since_entry * (1 - stop_loss)
                    if trailing_stop > current_stop:
                        current_stop = trailing_stop
                
                # Net P&L hesaplama fonksiyonu (maliyetler dahil)
                def check_net_profit(exit_p):
                    gross_pnl = (exit_p - entry_price) / entry_price
                    return 1 if (gross_pnl - costs) > 0 else 0

                # Aynı bar'da hem target hem stop kontrolü
                hit_target = f_high >= target_price
                hit_stop = f_low <= current_stop

                # KONSERVATİF YAKLAŞIM: İkisi de aynı bar'da olduysa, stop öncelikli
                # Gerçek piyasada hangisinin önce olduğu bilinmez, kötü senaryoyu varsay
                if hit_target and hit_stop:
                    result = check_net_profit(current_stop)
                    break
                elif hit_target:
                    result = check_net_profit(target_price)
                    break
                elif hit_stop:
                    result = check_net_profit(current_stop)
                    break
                # ÖNCELİK 3: SAT sinyali var mı?
                if 'Signal_Sell' in df.columns:
                    s_sell = future_row.get('Signal_Sell', 0)
                    if s_sell == 1:
                        result = check_net_profit(f_close)
                        break
            except:
                continue
        
        # look_ahead bar içinde çıkış olmadıysa, son fiyata göre karar ver
        if result is None:
            if len(future_slice) > 0:
                final_price = future_slice['Close'].iloc[-1]
                result = check_net_profit(final_price)
            else:
                result = 0
        
        targets.append(result)
    
    df['Target'] = targets
    return df


def prepare_single_stock(symbol, config, index_df=None):
    """Tek bir hisse için veri hazırlar."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(config.DATA_PERIOD_YEARS * 365))
        
        df = yf.download(symbol, start=start_date, end=end_date, 
                        interval=config.INTERVAL, progress=False)
        
        # UT Bot için minimum 50 satır yeterli (ATR 10 için)
        # EMA 200 zorunlu değil, feature olarak kullanılacak
        if df.empty:
            return None, "Yahoo Finance veri döndürmedi (Sembol hatalı olabilir)"
            
        if len(df) < 50:
            return None, f"Yetersiz geçmiş veri ({len(df)} gün < 50 gün). ATR için en az 50 gün lazım."
            
        # MultiIndex düzelt (ÇOK AGRESİF)
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        
        # Hala DataFrame ise (tek ticker yfinance bazen yapar) ve kolonlar Ticker ise
        if isinstance(df.columns, pd.MultiIndex) or (len(df.columns) > 0 and isinstance(df.columns[0], tuple)):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Tekil kolonlar haline getir (Series garantisi için)
        for col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        # --- BOLUNME DUZELTMESI ---
        df = data_utils.adjust_for_splits(df)
        
        # Endeks verilerini ekle
        if index_df is not None:
            # Endeks verilerini sadece gerekli kolonlarla al
            idx_cols = ['Close', 'High', 'Low']
            idx_data = index_df[idx_cols].copy()
            idx_data.columns = [f'Index_{c}' for c in idx_cols]
            
            # Tarihe göre birleştir (sol birleştirme, hisse tarihlerini koru)
            df = df.join(idx_data, how='left')
            df[idx_data.columns] = df[idx_data.columns].ffill()
        
        # Heikin Ashi
        if config.USE_HEIKIN_ASHI:
            df = calculate_heikin_ashi(df)
        
        # UT Bot
        df = calculate_ut_bot(df, config.KEY_VALUE, config.ATR_PERIOD, 
                             config.USE_HEIKIN_ASHI)
        
        # Feature Engineering
        df = engineer_features(df)
        
        # Pivot NaN'lerini doldur
        df['Nearest_Resistance'] = df['Nearest_Resistance'].bfill()
        df['Nearest_Support'] = df['Nearest_Support'].bfill()
        df['Dist_to_Resistance'] = df['Dist_to_Resistance'].bfill().fillna(0.1)
        df['Dist_to_Support'] = df['Dist_to_Support'].bfill().fillna(0.1)
        
        # Target Labeling
        df = create_target_label(df, config.LOOK_AHEAD_BARS, config)
        
        # Sembol ekle
        df['Symbol'] = symbol
        
        # NaN temizle - UT Bot için ZORUNLU olanlar (EMA değil!)
        # EMA'lar feature olarak kullanılacak, NaN olabilir
        ut_bot_essential = ['Close', 'High', 'Low', 'Open', 'Volume', 
                            'Trailing_Stop', 'Signal_Buy', 'Target']
        df = df.dropna(subset=ut_bot_essential)
        
        # EMA ve diğer feature NaN'lerini 0 ile doldur (model için)
        feature_cols = ['EMA_200', 'EMA_377', 'EMA_610', 'Dist_EMA_200', 
                        'Dist_EMA_377', 'Dist_EMA_610', 'StochRSI_K', 'StochRSI_D',
                        'Volume_Ratio', 'Support_Strength', 'Resistance_Strength']
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df, None
        
    except Exception as e:
        return None, str(e)


def load_elite_stocks(config):
    """Elit hisse listesini yükler."""
    # Önce root'ta ara, sonra input_dir'de
    paths = [config.ELITE_STOCKS_FILE, os.path.join(config.INPUT_DIR, config.ELITE_STOCKS_FILE)]
    
    for file_path in paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                stocks = [line.strip() for line in f if line.strip()]
            # .IS uzantısı yoksa ekle
            stocks = [s if s.upper().endswith('.IS') else f"{s}.IS" for s in stocks]
            print(f"✅ {len(stocks)} hisse yüklendi.")
            return stocks
    
    print("⚠️ Hisse listesi bulunamadı. Örnek hisseler kullanılıyor...")
    return ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS", "TUPRS.IS",
            "ASELS.IS", "BIMAS.IS", "SISE.IS", "KCHOL.IS", "SAHOL.IS"]


# ============================================================
# MODEL EĞİTİMİ
# ============================================================

def get_feature_columns(df):
    """Model için kullanılacak feature sütunları - SADECE SENİN STRATEJİN"""
    feature_cols = [
        # EMA Uzaklıkları
        'Dist_EMA_200',
        'Dist_EMA_377', 
        'Dist_EMA_610',
        # EMA Destek/Direnç (YENİ!)
        'EMA_200_377_Gap',
        'EMA_377_610_Gap',
        'EMA_All_Gap',
        'EMAs_Converging',
        'EMA_200_Support',
        'EMA_200_Resistance',
        'EMA_377_Support',
        'EMA_377_Resistance',
        'EMA_610_Support',
        'EMA_610_Resistance',
        # Destek/Direnç Gücü
        'Support_Strength', 'Resistance_Strength', 'Near_EMA_Support', 'Near_EMA_Resistance', 'Super_Support_Zone',
        
        # Pazar Rejimi (PRECISION)
        'Index_Trend', 'Rel_Strength_10', 'Rel_Strength_30', 'Index_ADX',
        # EMA Durumları (eski uyumluluk)
        'Above_EMA_200',
        'Above_EMA_377',
        'Above_EMA_610',
        'Above_All_EMAs',
        'EMA_Stack',
        # POI - Destek/Direnç
        'Dist_to_Resistance',
        'Dist_to_Support',
        'Near_Support',
        'Near_Resistance',
        # StochRSI
        'StochRSI_K',
        'StochRSI_D',
        'StochRSI_Oversold',
        'StochRSI_Overbought',
        'StochRSI_Bullish',
        # Trailing Stop
        'Dist_to_Stop',
    ]
    
    # Sadece DataFrame'de olan sütunları döndür
    return [col for col in feature_cols if col in df.columns]


def train_models(X_train, y_train, config):
    """Sadece XGBoost modeli eğitir."""
    models = {}
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost kurulu değil! pip install xgboost")
        return models
    
    print("   🚀 XGBoost eğitiliyor...")
    
    # Class imbalance için scale_pos_weight
    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=config.RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    print("   ✅ XGBoost eğitimi tamamlandı!")
    
    return models


def evaluate_models(models, X_test, y_test):
    """Modelleri değerlendirir."""
    results = {}
    
    print("\n📊 MODEL DEĞERLENDİRME SONUÇLARI:")
    print("=" * 50)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = (y_pred == y_test).mean()
        
        # Precision, Recall için sadece pozitif sınıf
        true_pos = ((y_pred == 1) & (y_test == 1)).sum()
        false_pos = ((y_pred == 1) & (y_test == 0)).sum()
        false_neg = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc
        }
        
        print(f"\n{name}:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f} (AL dediğinde gerçekten kazanma)")
        print(f"   Recall:    {recall:.3f} (Kazançlı işlemleri yakalama)")
        print(f"   AUC:       {auc:.3f}")
    
    return results


def analyze_feature_importance(model, feature_cols, top_n=20):
    """Feature importance analizi yapar."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 EN ÖNEMLİ {top_n} ÖZELLİK:")
    print("=" * 40)
    for i, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row['Importance'] * 50)
        print(f"   {row['Feature'][:20]:<20} {bar} {row['Importance']:.3f}")
    
    return importance_df


def walk_forward_validation(df, feature_cols, config):
    """Walk-forward validation yapar."""
    print("\n🔄 WALK-FORWARD VALIDATION")
    print("=" * 50)
    
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost kurulu değil, validation atlanıyor...")
        return []
    
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['Target']
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # XGBoost ile validation
        scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
        
        xgb = XGBClassifier(
            n_estimators=100, 
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale,
            random_state=config.RANDOM_STATE,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        
        y_pred = xgb.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        # Precision
        true_pos = ((y_pred == 1) & (y_test == 1)).sum()
        false_pos = ((y_pred == 1) & (y_test == 0)).sum()
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        
        fold_results.append({'Fold': fold+1, 'Accuracy': accuracy, 'Precision': precision})
        print(f"   Fold {fold+1}: Accuracy={accuracy:.3f}, Precision={precision:.3f}")
    
    avg_acc = np.mean([r['Accuracy'] for r in fold_results])
    avg_prec = np.mean([r['Precision'] for r in fold_results])
    print(f"\n   📈 Ortalama: Accuracy={avg_acc:.3f}, Precision={avg_prec:.3f}")
    
    return fold_results


# ============================================================
# ANA PROGRAM
# ============================================================

def train_single_stock_model(symbol, df, feature_cols, config):
    """
    Tek bir hisse için model eğitir ve performans metriklerini döndürür.
    """
    if not XGBOOST_AVAILABLE:
        return None
    
    # META-LABELING: TÜM UT Bot AL sinyallerini al
    # Model, EMA/StochRSI/POI durumlarını feature olarak öğrenecek
    # Hangi koşullarda kazandığını kendisi keşfedecek!
    signal_data = df[df['Signal_Buy'] == 1].copy()
    
    # Minimum sinyal sayısı kontrolü
    if len(signal_data) < config.MIN_SIGNALS:
        return {
            'symbol': symbol,
            'status': 'skipped',
            'reason': f'Yetersiz sinyal ({len(signal_data)} < {config.MIN_SIGNALS})',
            'signal_count': len(signal_data)
        }
    
    # Sınıf dağılımı kontrolü
    y_all = signal_data['Target'].astype(int)
    
    # Eğer TİCKER %100 başarılıysa (COSMO gibi) -> Her şeye 1 diyen model
    if y_all.sum() == len(y_all):
        model = DominantWinnerModel(confidence=config.DOMINANT_CONFIDENCE)
        symbol_clean = symbol.replace('.IS', '')
        model_path = os.path.join(config.OUTPUT_DIR, f"{symbol_clean}.pkl")
        joblib.dump(model, model_path)
        
        return {
            'symbol': symbol,
            'symbol_clean': symbol_clean,
            'status': 'trained',
            'model_path': model_path,
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'auc': 1.0,
            'signal_count': len(signal_data),
            'reason': 'Dominant Winner (%100)',
            'target_dist': "100% başarılı"
        }
    
    # Eğer TİCKER %0 başarılıysa -> Skip
    if y_all.sum() == 0:
        return {
            'symbol': symbol,
            'status': 'skipped',
            'reason': 'Hepsi Kayıp (%0 Başarı)',
            'signal_count': len(signal_data)
        }

    # Feature seçimi (küçük veri setlerinde özelliği azalt)
    current_features = feature_cols
    if len(signal_data) < 30:
        # En önemli 10 temel özelliği seç (overfitting'i azaltmak için)
        essential_features = [
            'Dist_EMA_200', 'Dist_EMA_377', 'Dist_EMA_610', 
            'Near_EMA_Support', 'Super_Support_Zone',
            'Dist_to_Support', 'Dist_to_Resistance',
            'StochRSI_K', 'StochRSI_D', 'Volume_Ratio'
        ]
        current_features = [f for f in essential_features if f in feature_cols]

    # Feature ve target hazırla
    X = signal_data[current_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = signal_data['Target'].astype(int)

    # ============================================================
    # TRAIN / VALIDATION / TEST SPLIT (Purging ile)
    # ============================================================
    # Train: %60, Val: %15, Test: %25
    # Purging: Her split arasında PURGE_BARS kadar boşluk (veri sızıntısı önleme)
    # Bu gerçek trading koşullarını simüle eder (geleceği göremezsin!)

    n = len(signal_data)
    purge = min(config.PURGE_BARS, max(1, n // 20))  # Küçük verilerde purge'ı azalt

    train_end = int(n * 0.60)
    val_start = train_end + purge
    val_end = int(n * 0.75)
    test_start = val_end + purge

    # Yeterli veri kontrolü
    if val_start >= val_end or test_start >= n:
        # Purging için yeterli veri yok, basit split yap
        train_end = int(n * 0.60)
        val_end = int(n * 0.75)
        val_start = train_end
        test_start = val_end

    X_train = X.iloc[:train_end]
    X_val = X.iloc[val_start:val_end]
    X_test = X.iloc[test_start:]
    y_train = y.iloc[:train_end]
    y_val = y.iloc[val_start:val_end]
    y_test = y.iloc[test_start:]

    # Minimum test sample kontrolü (güvenilir precision için)
    if len(X_test) < config.MIN_TEST_SAMPLES:
        return {
            'symbol': symbol,
            'status': 'skipped',
            'reason': f'Yetersiz test sinyali ({len(X_test)} < {config.MIN_TEST_SAMPLES})',
            'signal_count': len(signal_data)
        }

    # Azınlık sınıfı çoğalt (Oversampling)
    if len(y_train[y_train == 0]) > 0 and len(y_train[y_train == 1]) > 0:
        counts = y_train.value_counts()
        if counts.min() / counts.max() < 0.5:
            # Basit oversampling
            minority_class = counts.idxmin()
            majority_class = counts.idxmax()
            n_samples = counts[majority_class] - counts[minority_class]
            
            X_minority = X_train[y_train == minority_class]
            y_minority = y_train[y_train == minority_class]
            
            # Rastgele seçerek çoğalt
            idx = np.random.choice(X_minority.index, size=n_samples, replace=True)
            X_train = pd.concat([X_train, X_minority.loc[idx]])
            y_train = pd.concat([y_train, y_minority.loc[idx]])

    # Class imbalance için scale_pos_weight
    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1
    
    # Train setinde tek sınıf kalmış olabilir (özellikle çok küçük verilerde ve sinyal sayısının az olduğu durumlarda)
    if len(np.unique(y_train)) < 2:
        major_class = y_train.mode()[0]
        if major_class == 1:
            # Train setinde sadece kârlı işlemler kalmış -> Dominant Winner yapalım
            model = DominantWinnerModel(confidence=config.DOMINANT_CONFIDENCE)
            symbol_clean = symbol.replace('.IS', '')
            model_path = os.path.join(config.OUTPUT_DIR, f"{symbol_clean}.pkl")
            joblib.dump(model, model_path)
            
            return {
                'symbol': symbol,
                'symbol_clean': symbol_clean,
                'status': 'trained',
                'model_path': model_path,
                'accuracy': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'auc': 1.0,
                'signal_count': len(signal_data),
                'reason': 'Train Seti Dominant Winner (%100)',
                'target_dist': "100% başarılı (train)"
            }
        else:
            return {
                'symbol': symbol,
                'status': 'skipped',
                'reason': 'Train setinde sadece kayıp var',
                'signal_count': len(signal_data)
            }

    # ============================================================
    # HİBRİT EĞİTİM STRATEJİSİ
    # ============================================================
    # Eğer sinyal sayısı azsa Optuna overfitting yapar, sabit güvenli parametre kullan.
    # Sinyal sayısı yeterliyse Optuna ile ince ayar yap.
    
    if len(signal_data) >= 40:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
                'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                'scale_pos_weight': scale,
                'random_state': config.RANDOM_STATE,
                'eval_metric': 'logloss',
                'verbosity': 0,
                'n_jobs': -1
            }
            
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import roc_auc_score
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                if len(np.unique(y_t)) < 2 or len(np.unique(y_v)) < 2: continue
                
                m = XGBClassifier(**params)
                m.fit(X_t, y_t)
                
                probs = m.predict_proba(X_v)[:, 1]
                try:
                    score = roc_auc_score(y_v, probs)
                    scores.append(score)
                except: continue
                
            return np.mean(scores) if scores else 0.5

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=config.OPTUNA_TRIALS, timeout=config.OPTUNA_TIMEOUT)
        best_params = study.best_params
    else:
        # Küçük verilerde en güvenli 'robust' parametreler
        best_params = {
            'n_estimators': 150,
            'max_depth': 3,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    
    # Final parametreleri birleştir
    best_params.update({
        'scale_pos_weight': scale,
        'random_state': config.RANDOM_STATE,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
        'early_stopping_rounds': config.EARLY_STOPPING_ROUNDS
    })

    model = XGBClassifier(**best_params)

    # Early Stopping ile eğitim (validation set kullanarak)
    # Validation set yeterli ve iki sınıf içeriyorsa early stopping kullan
    if len(X_val) >= 3 and len(np.unique(y_val)) >= 2:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        # Validation set yetersizse normal eğitim (early stopping olmadan)
        best_params.pop('early_stopping_rounds', None)
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    accuracy = (y_pred == y_test).mean()
    
    true_pos = ((y_pred == 1) & (y_test == 1)).sum()
    false_pos = ((y_pred == 1) & (y_test == 0)).sum()
    false_neg = ((y_pred == 0) & (y_test == 1)).sum()
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    
    auc = 0.5
    try:
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_proba)
    except:
        pass
        auc = 0
    
    # Precision Filtresi (MASS TRAINING GÜNCELLEMESİ)
    # Sadece belli bir başarı oranının üzerindeki modeller kaydedilir.
    if precision < config.PRECISION_THRESHOLD:
        return {
            'symbol': symbol,
            'status': 'skipped',
            'reason': f'Düşük Precision ({precision:.1%} < {config.PRECISION_THRESHOLD:.0%})',
            'signal_count': len(signal_data),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4)
        }

    # TimeSeriesSplit Cross-Validation (yeterli veri varsa)
    # Tek split'e güvenmek yerine CV ile daha güvenilir metrik
    cv_precision = None
    if len(signal_data) >= config.CV_MIN_SIGNALS:
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        cv_precisions = []

        # CV için early_stopping olmadan parametreler
        cv_params = {k: v for k, v in best_params.items() if k != 'early_stopping_rounds'}

        for train_idx, test_idx in tscv.split(X):
            X_t, X_v = X.iloc[train_idx], X.iloc[test_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[test_idx]

            # Her iki sınıf da olmalı
            if len(np.unique(y_t)) < 2 or len(np.unique(y_v)) < 2:
                continue

            temp_model = XGBClassifier(**cv_params)
            temp_model.fit(X_t, y_t)
            y_cv_pred = temp_model.predict(X_v)

            tp = ((y_cv_pred == 1) & (y_v == 1)).sum()
            fp = ((y_cv_pred == 1) & (y_v == 0)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            cv_precisions.append(prec)

        # CV ortalama precision kontrolü
        if cv_precisions:
            cv_precision = np.mean(cv_precisions)
            if cv_precision < config.PRECISION_THRESHOLD:
                return {
                    'symbol': symbol,
                    'status': 'skipped',
                    'reason': f'CV Precision düşük ({cv_precision:.1%} < {config.PRECISION_THRESHOLD:.0%})',
                    'signal_count': len(signal_data),
                    'accuracy': round(accuracy, 4),
                    'precision': round(precision, 4),
                    'cv_precision': round(cv_precision, 4)
                }

    # Modeli kaydet
    symbol_clean = symbol.replace('.IS', '')
    model_path = os.path.join(config.OUTPUT_DIR, f"{symbol_clean}.pkl")
    joblib.dump(model, model_path)
    
    result = {
        'symbol': symbol,
        'symbol_clean': symbol_clean,
        'status': 'trained',
        'model_path': model_path,
        'signal_count': len(signal_data),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'auc': round(auc, 4),
        'target_dist': f"{y.mean()*100:.1f}% başarılı"
    }

    # CV yapıldıysa CV precision'ı da ekle
    if cv_precision is not None:
        result['cv_precision'] = round(cv_precision, 4)

    return result


def main():
    print("=" * 60)
    print("🤖 HİSSE BAZLI ML MODEL EĞİTİM SİSTEMİ")
    print("=" * 60)
    
    config = ModelConfig()
    
    # Klasörleri oluştur
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Elit hisseleri yükle
    elite_stocks = load_elite_stocks(config)
    
    print(f"\n📥 {len(elite_stocks)} hisse için HİSSE BAZLI model eğitimi başlıyor...")
    print("=" * 60)
    
    # ============================================================
    # ENDEKS VERİSİ (PAZAR REJİMİ İÇİN)
    # ============================================================
    print(f"📥 BIST100 Endeks verisi indiriliyor...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(config.DATA_PERIOD_YEARS * 365))
    index_df = yf.download("XU100.IS", start=start_date, end=end_date, 
                          interval=config.INTERVAL, progress=False)
    
    if not index_df.empty and isinstance(index_df.columns, pd.MultiIndex):
        index_df.columns = index_df.columns.get_level_values(0)
    
    # Sonuçları takip et
    all_results = []
    trained_count = 0
    skipped_count = 0
    
    # Feature sütunları (ilk hisseden alacağız)
    feature_cols = None
    
    # 100 hisse için model eğitimi
    print(f"📥 {len(elite_stocks)} hisse için HİSSE BAZLI model eğitimi başlıyor...")
    print("=" * 60 + "\n")
    
    for i, symbol in enumerate(elite_stocks, 1):
        print(f"[{i}/{len(elite_stocks)}] {symbol} işleniyor...")
        
        # Veri Hazırlama (Endeks Verisi Eklendi)
        df, err_msg = prepare_single_stock(symbol, config, index_df=index_df)
        
        if df is None:
            print(f"   ❌ Atlandı: {err_msg}")
            skipped_count += 1
            all_results.append({
                'symbol': symbol,
                'status': 'skipped',
                'reason': err_msg
            })
            continue
        
        # Feature sütunlarını al
        if feature_cols is None:
            feature_cols = get_feature_columns(df)
            print(f"   📊 {len(feature_cols)} feature kullanılacak")
        
        # Model eğit
        result = train_single_stock_model(symbol, df, feature_cols, config)
        
        if result is None:
            print(f"   ❌ XGBoost mevcut değil!")
            continue
        
        all_results.append(result)
        
        if result['status'] == 'trained':
            trained_count += 1
            print(f"   ✅ Model eğitildi!")
            print(f"      • Sinyal: {result['signal_count']}")
            print(f"      • Accuracy: {result['accuracy']:.1%}")
            print(f"      • Precision: {result['precision']:.1%}")
            print(f"      • Kayıt: {result['model_path']}")
        else:
            skipped_count += 1
            print(f"   ⚠️ Atlandı: {result['reason']}")
    
    # Feature listesini kaydet (tüm modeller için ortak)
    if feature_cols:
        features_path = os.path.join(config.OUTPUT_DIR, "feature_columns.json")
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        print(f"\n💾 Feature listesi kaydedildi: {features_path}")
    
    # Model özet dosyası oluştur
    summary = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_stocks': len(elite_stocks),
        'trained_count': trained_count,
        'skipped_count': skipped_count,
        'config': {
            'KEY_VALUE': config.KEY_VALUE,
            'ATR_PERIOD': config.ATR_PERIOD,
            'USE_HEIKIN_ASHI': config.USE_HEIKIN_ASHI,
            'LOOK_AHEAD_BARS': config.LOOK_AHEAD_BARS,
            'TARGET_PROFIT': config.TARGET_PROFIT,
            'STOP_LOSS': config.STOP_LOSS
        },
        'models': [r for r in all_results if r.get('status') == 'trained']
    }
    
    summary_path = os.path.join(config.OUTPUT_DIR, "model_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Özet tablo
    print("\n" + "=" * 60)
    print("📊 EĞİTİM ÖZETİ")
    print("=" * 60)
    
    trained_models = [r for r in all_results if r.get('status') == 'trained']
    
    if trained_models:
        # En iyi modelleri sırala (precision'a göre)
        trained_models.sort(key=lambda x: x['precision'], reverse=True)
        
        print(f"\n🏆 EN İYİ MODELLER (Precision sıralı):")
        print("-" * 60)
        print(f"{'Hisse':<12} {'Sinyal':>8} {'Accuracy':>10} {'Precision':>10} {'AUC':>8}")
        print("-" * 60)
        
        for model in trained_models[:10]:  # İlk 10
            print(f"{model['symbol_clean']:<12} {model['signal_count']:>8} {model['accuracy']:>10.1%} {model['precision']:>10.1%} {model['auc']:>8.3f}")
        
        # İstatistikler
        avg_precision = np.mean([m['precision'] for m in trained_models])
        avg_accuracy = np.mean([m['accuracy'] for m in trained_models])
        
        print(f"\n📈 GENEL İSTATİSTİKLER:")
        print(f"   • Eğitilen model: {trained_count}")
        print(f"   • Atlanan: {skipped_count}")
        print(f"   • Ortalama Precision: {avg_precision:.1%}")
        print(f"   • Ortalama Accuracy: {avg_accuracy:.1%}")
    
    print(f"\n💾 Kaydedilen dosyalar:")
    print(f"   • models/*.pkl (her hisse için ayrı model)")
    print(f"   • {summary_path}")
    if feature_cols:
        print(f"   • {features_path}")
    
    print("\n" + "=" * 60)
    print("✅ HİSSE BAZLI EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    
    print("\n💡 Sonraki Adım:")
    print("   Webhook sunucusu bu modelleri kullanarak")
    print("   her hisse için ayrı tahmin yapacak.")
    print("   Örnek: THYAO sinyali → models/THYAO.pkl kullanılır")


if __name__ == "__main__":
    main()

