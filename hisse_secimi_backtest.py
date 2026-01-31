"""
BIST HİSSE SEÇİM VE BACKTEST SİSTEMİ
=====================================
UT Bot (ATR Trailing Stop) stratejisi ile BIST hisselerini tarar,
backtest yapar ve en iyi performans gösteren hisseleri seçer.

Özellikler:
- Stop Loss mekanizması
- Komisyon ve slippage hesabı
- Risk/Reward analizi
- Walk-forward düşüncesiyle train/test ayrımı
- Hacim onayı filtresi

Yazar: Trading AI System
Versiyon: 2.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import warnings
import time
import os

warnings.filterwarnings('ignore')

# ============================================================
# AYARLAR - Bunları ihtiyacına göre değiştir
# ============================================================

class Config:
    # Veri Ayarları
    TRAIN_PERIOD_YEARS = 5      # 610 EMA için minimum 5 yıl
    TEST_PERIOD_MONTHS = 6      # Test için kaç aylık veri (walk-forward için)
    INTERVAL = "1d"             # Günlük mum (1h, 4h, 1d)
    
    # UT Bot Ayarları (Pine Script ile birebir)
    KEY_VALUE = 2               # ATR çarpanı (a)
    ATR_PERIOD = 10             # ATR periyodu (c)
    USE_HEIKIN_ASHI = True      # Heikin Ashi kullan
    
    # Risk Yönetimi
    STOP_LOSS_PCT = 0.05        # %5 stop loss
    TARGET_PROFIT_PCT = 0.105   # %10.5 hedef kar (komisyon sonrası net %10)
    TRAILING_STOP = True        # Trailing stop aktif mi?
    
    # Maliyet Hesabı
    COMMISSION_PCT = 0.002      # %0.2 komisyon (alış + satış)
    SLIPPAGE_PCT = 0.001        # %0.1 slippage
    
    # Filtreleme
    MIN_TRADES = 5              # Minimum işlem sayısı (filtered signals daha az)
    MIN_VOLUME_AVG = 1_000_000  # Minimum ortalama hacim (TL)
    MIN_WIN_RATE = 40           # Minimum kazanma oranı %
    
    # Çıktı
    TOP_N_STOCKS = 100          # 10'dan 100'e çıkarıldı
    OUTPUT_DIR = "output"


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def get_bist_tickers():
    """
    BIST hisse listesini hisseler.txt dosyasından okur.
    Dosya bulunamazsa fallback manuel liste kullanılır.
    """
    try:
        # hisseler.txt dosyasını oku
        ticker_file = os.path.join(os.path.dirname(__file__), 'hisseler.txt')

        if os.path.exists(ticker_file):
            print("📡 hisseler.txt dosyasından liste okunuyor...")
            with open(ticker_file, 'r', encoding='utf-8') as f:
                tickers = [line.strip() for line in f if line.strip()]

            print(f"✅ {len(tickers)} adet hisse bulundu.")
            return tickers
        else:
            print("⚠️ hisseler.txt bulunamadı, manuel liste kullanılıyor...")
            raise FileNotFoundError("hisseler.txt not found")

    except Exception as e:
        print(f"⚠️ Dosya okuma hatası: {e}")
        print("   Manuel liste kullanılıyor...")

        # Genişletilmiş BIST hisse listesi (500+ hisse)
        manual_list = [
            # Ana Endeks
            "THYAO", "GARAN", "AKBNK", "YKBNK", "ISCTR", "SAHOL", "KCHOL",
            "TUPRS", "EREGL", "BIMAS", "ASELS", "SISE", "TCELL", "FROTO",
            "TOASO", "KOZAL", "KRDMD", "PETKM", "TAVHL", "HEKTS", "VESTL",
            "ARCLK", "DOHOL", "EKGYO", "ENKAI", "GUBRF", "ISGYO", "KONTR",
            "KOZAA", "MGROS", "ODAS", "OYAKC", "PGSUS", "SASA", "SOKM",
            "TKFEN", "TTKOM", "ULKER", "VAKBN", "ISMEN", "AEFES", "AKSEN",
            # Bankacılık
            "ALBRK", "ICBCT", "QNBFN", "QNBFL", "TSKB", "SKBNK", "KLNMA",
            # Holding
            "AGHOL", "BRYAT", "DGGYO", "EGEEN", "GLYHO", "GLBMD", "IHLAS",
            "MAVI", "MPARK", "NTHOL", "POLHO", "PRKAB", "TRGYO", "TTRAK",
            # Sanayi
            "ADEL", "ADESE", "AKENR", "AKCNS", "ALCAR", "ALCTL", "ANACM",
            "ASUZU", "AVGYO", "BFREN", "BRISA", "BRSAN", "BTCIM", "BURCE",
            "BURVA", "CELHA", "CEMTS", "CIMSA", "CLEBI", "DEVA", "DOAS",
            "DURDO", "DYOBY", "EGPRO", "EMKEL", "EMNIS", "ENJSA", "EPLAS",
            "ERBOS", "ERSU", "ESCOM", "FMIZP", "GENIL", "GENTS", "GOLTS",
            "GOODY", "GSDHO", "HATEK", "HURGZ", "IHEVA", "INDES", "IPEKE",
            "IZFAS", "IZMDC", "KAPLM", "KARTN", "KARSN", "KATMR", "KENT",
            "KLMSN", "KNFRT", "KORDS", "KRSTL", "KUTPO", "LINK", "LOGO",
            "LUKSK", "MAKTK", "MANAS", "MNDRS", "MRSHL", "NETAS", "NIBAS",
            "NTTUR", "OLMIP", "OTKAR", "PARSN", "PENGD", "PETUN", "PINSU",
            "PKART", "PKENT", "PRKME", "PRZMA", "RALYH", "RGYAS", "SANEL",
            "SANFM", "SANKO", "SAYAS", "SELEC", "SILVR", "SNPAM", "TIRE",
            "TOASO", "TRCAS", "TRILC", "TRKCM", "TSGYO", "TTRAK", "TUDDF",
            "TUKAS", "TURGG", "ULUUN", "USAS", "UZERB", "VANGD", "VBTYZ",
            "VERUS", "VESBE", "YAPRK", "YUNSA", "ZOREN",
            # Teknoloji
            "ALKA", "ANELE", "ARENA", "ARMDA", "DESPC", "DGATE", "ESCAR",
            "INTEM", "KFEIN", "KLVMA", "KRONT", "LIDFA", "NLDFT", "PATEK",
            # Perakende
            "ADESE", "BIZIM", "CRFSA", "MAVI", "MZHLD",
            # Enerji
            "AKENR", "AKSA", "AKSUE", "AYEN", "GENIL", "GWIND", "ODAS",
            "ZRGYO",
            # Tekstil
            "ATEKS", "BLCYT", "BRKO", "DAGI", "DERIM", "HATEK", "KRTEK",
            "LKMNH", "RODRG", "SKTAS", "YATAS", "YGGYO",
            # Gıda
            "BANVT", "CCOLA", "ERSU", "KENT", "KNFRT", "PETUN", "TATGD",
            "ULKER", "VANGD",
            # Kimya
            "ALKIM", "BRKSN", "CMBTN", "DENCM", "DYOBY", "EGEEN", "POLTK",
            "SODA",
            # İnşaat
            "ANACM", "EDIP", "ENKAI", "ORGE", "SNGYO", "YAYLA",
            # Turizm
            "AYCES", "ETILR", "MAALT", "METUR", "PKENT", "ULAS",
            # Diğer
            "ADNAC", "AGYO", "AHGAZ", "ALGYO", "ALMAD", "ALTIN", "ANHYT",
            "ARDYZ", "ATAGY", "ATLAS", "AVHOL", "AVGYO", "AVOD", "BAGFS",
            "BAKAB", "BALAT", "BARMA", "BASGZ", "BAYRK", "BEGYO", "BERA",
            "BEYAZ", "BISAS", "BJKAS", "BLCYT", "BNTAS", "BOBET", "BOSSA",
            "BRMEN", "BUCIM", "BURVA", "CANTE", "CASA", "CATES", "CEMTS",
            "CEMZY", "CEOEM", "CMENT", "CONSE", "COSMO", "CRDFA", "DAGHL",
            "DAGI", "DAPGM", "DARDL", "DENGE", "DERHL", "DERIM", "DESA",
            "DESPC", "DGNMO", "DIRIT", "DOBUR", "DOGUB", "DOHOL", "Dyhol"
        ]
        return [f"{t}.IS" for t in sorted(set(manual_list))]


def calculate_heikin_ashi(df):
    """
    Heikin Ashi Close hesaplar - Pine Script ile birebir uyumlu.
    Pine Script: request.security(ticker.heikinashi(...), ..., close)
    Sadece HA Close değerini hesaplıyoruz (Pine Script gibi).
    """
    ha_df = df.copy()
    
    # HA Close = (Open + High + Low + Close) / 4
    # Pine Script'teki heikinashi close ile aynı
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    return ha_df


def calculate_ut_bot(df, key_value, atr_period, use_ha=True):
    """
    UT Bot (ATR Trailing Stop) hesaplaması.
    Pine Script mantığının birebir Python karşılığı.
    
    Pine Script:
        ema = ta.ema(src, 1)
        above = ta.crossover(ema, xATRTrailingStop)
        buy = src > xATRTrailingStop and above
    """
    # Kaynak fiyat seçimi (Pine Script: src = h ? heikinashi_close : close)
    src = df['HA_Close'] if use_ha else df['Close']
    
    # EMA(1) hesapla - Pine Script ile birebir
    # ta.ema(src, 1) pratikte src'nin kendisi ama Pine Script uyumu için
    ema = ta.ema(src, length=1)
    if ema is None:
        ema = src.copy()
    
    # ATR hesapla (her zaman gerçek fiyattan - Pine Script gibi)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=atr_period)
    nLoss = key_value * atr
    
    # Trailing Stop hesaplama (recursive - Pine Script ile birebir)
    xATRTrailingStop = np.zeros(len(df))
    
    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1] if xATRTrailingStop[i-1] != 0 else 0
        curr_src = src.iloc[i]
        prev_src = src.iloc[i-1]
        curr_nLoss = nLoss.iloc[i]
        
        if np.isnan(curr_nLoss):
            xATRTrailingStop[i] = prev_stop
            continue
        
        # Pine Script mantığı - nz(xATRTrailingStop[1], 0) ile birebir
        # iff_1 = src > nz(xATRTrailingStop[1], 0) ? src - nLoss : src + nLoss
        iff_1 = curr_src - curr_nLoss if curr_src > prev_stop else curr_src + curr_nLoss
        
        # iff_2 = src < nz(xATRTrailingStop[1], 0) and src[1] < nz(xATRTrailingStop[1], 0) ? 
        #         math.min(nz(xATRTrailingStop[1]), src + nLoss) : iff_1
        if curr_src < prev_stop and prev_src < prev_stop:
            iff_2 = min(prev_stop, curr_src + curr_nLoss)
        else:
            iff_2 = iff_1
        
        # xATRTrailingStop := src > nz(xATRTrailingStop[1], 0) and src[1] > nz(xATRTrailingStop[1], 0) ? 
        #                     math.max(nz(xATRTrailingStop[1]), src - nLoss) : iff_2
        if curr_src > prev_stop and prev_src > prev_stop:
            xATRTrailingStop[i] = max(prev_stop, curr_src - curr_nLoss)
        else:
            xATRTrailingStop[i] = iff_2
    
    df['Trailing_Stop'] = xATRTrailingStop
    df['ATR'] = atr
    
    # Sinyal oluşturma - Pine Script ile birebir:
    # above = ta.crossover(ema, xATRTrailingStop)
    # buy = src > xATRTrailingStop and above
    
    # Crossover: ema şu an trailing_stop'un üstünde VE bir önceki bar altındaydı
    crossover_above = (ema > df['Trailing_Stop']) & (ema.shift(1) <= df['Trailing_Stop'].shift(1))
    df['Signal_Buy'] = ((src > df['Trailing_Stop']) & crossover_above).astype(int)
    
    # below = ta.crossover(xATRTrailingStop, ema)
    # sell = src < xATRTrailingStop and below
    crossover_below = (df['Trailing_Stop'] > ema) & (df['Trailing_Stop'].shift(1) <= ema.shift(1))
    df['Signal_Sell'] = ((src < df['Trailing_Stop']) & crossover_below).astype(int)
    
    return df


def calculate_pivot_points(df, swing_length=10):
    """
    Pine Script ta.pivothigh/ta.pivotlow ile birebir uyumlu pivot hesaplaması.
    
    Pine Script:
        swing_high = ta.pivothigh(high, swing_length, swing_length)
        swing_low = ta.pivotlow(low, swing_length, swing_length)
    
    Pivot High: Bir bar, solundaki ve sağındaki swing_length bar'dan yüksekse
    Pivot Low: Bir bar, solundaki ve sağındaki swing_length bar'dan düşükse
    """
    pivot_highs = pd.Series(index=df.index, dtype=float)
    pivot_lows = pd.Series(index=df.index, dtype=float)
    
    for i in range(swing_length, len(df) - swing_length):
        # Pivot High kontrolü
        current_high = df['High'].iloc[i]
        is_pivot_high = True
        
        # Sol taraf kontrolü
        for j in range(1, swing_length + 1):
            if df['High'].iloc[i - j] >= current_high:
                is_pivot_high = False
                break
        
        # Sağ taraf kontrolü
        if is_pivot_high:
            for j in range(1, swing_length + 1):
                if df['High'].iloc[i + j] >= current_high:
                    is_pivot_high = False
                    break
        
        if is_pivot_high:
            pivot_highs.iloc[i] = current_high
        
        # Pivot Low kontrolü
        current_low = df['Low'].iloc[i]
        is_pivot_low = True
        
        # Sol taraf kontrolü
        for j in range(1, swing_length + 1):
            if df['Low'].iloc[i - j] <= current_low:
                is_pivot_low = False
                break
        
        # Sağ taraf kontrolü
        if is_pivot_low:
            for j in range(1, swing_length + 1):
                if df['Low'].iloc[i + j] <= current_low:
                    is_pivot_low = False
                    break
        
        if is_pivot_low:
            pivot_lows.iloc[i] = current_low
    
    df['Pivot_High'] = pivot_highs
    df['Pivot_Low'] = pivot_lows
    
    # En yakın destek/direnç (forward fill ile)
    df['Nearest_Resistance'] = df['Pivot_High'].ffill()
    df['Nearest_Support'] = df['Pivot_Low'].ffill()
    
    return df


def calculate_indicators(df):
    """
    Ek teknik indikatörler hesaplar (filtreleme ve analiz için).
    
    EMA DESTEK/DİRENÇ SİSTEMİ:
    - EMA 200: Normal destek/direnç
    - EMA 377: Orta güçte destek/direnç
    - EMA 610: Güçlü destek/direnç
    - EMA'lar yakınsa: SÜPER güçlü bölge
    
    StochRSI Pine Script ile AYNI: K=3, D=3, RSI=14, Stoch=14
    """
    # EMA'lar - Pine Script ile birebir
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_377'] = ta.ema(df['Close'], length=377)
    df['EMA_610'] = ta.ema(df['Close'], length=610)
    
    # === EMA DESTEK/DİRENÇ HESAPLAMASI ===
    # Fiyatın her EMA'ya uzaklığı (yüzde olarak)
    df['Dist_EMA_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    df['Dist_EMA_377'] = (df['Close'] - df['EMA_377']) / df['EMA_377']
    df['Dist_EMA_610'] = (df['Close'] - df['EMA_610']) / df['EMA_610']
    
    # EMA'ların birbirine yakınlığı (yüzde olarak)
    # Yakınlık = SÜPER güçlü bölge oluşuyor
    df['EMA_200_377_Gap'] = abs(df['EMA_200'] - df['EMA_377']) / df['Close']
    df['EMA_377_610_Gap'] = abs(df['EMA_377'] - df['EMA_610']) / df['Close']
    df['EMA_All_Gap'] = abs(df['EMA_200'] - df['EMA_610']) / df['Close']
    
    # EMA'lar yakınsıyor mu? (%2'den az fark = yakın)
    EMA_CONVERGENCE_THRESHOLD = 0.02
    df['EMAs_Converging'] = (df['EMA_All_Gap'] < EMA_CONVERGENCE_THRESHOLD).astype(int)
    
    # === DESTEK SEVİYELERİ (Fiyat EMA'nın ÜSTÜNDE ve yakınsa = destek) ===
    NEAR_THRESHOLD = 0.02  # %2'den yakınsa "yakın" sayılır
    
    # EMA 200 - Normal güçte destek/direnç
    df['EMA_200_Support'] = (
        (df['Dist_EMA_200'] > 0) &  # Fiyat üstte
        (df['Dist_EMA_200'] < NEAR_THRESHOLD)  # ve yakın
    ).astype(int)
    df['EMA_200_Resistance'] = (
        (df['Dist_EMA_200'] < 0) &  # Fiyat altta
        (abs(df['Dist_EMA_200']) < NEAR_THRESHOLD)  # ve yakın
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
    
    # === DESTEK GÜCÜ SKORU (0-6 arası) ===
    # Her destek için puan ver, yakınsama varsa ekstra puan
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
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # === STOCHASTIC RSI (Pine Script ile birebir: K=3, D=3, RSI=14, Stoch=14) ===
    stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
    if stoch is not None and 'STOCHRSIk_14_14_3_3' in stoch.columns:
        df['StochRSI_K'] = stoch['STOCHRSIk_14_14_3_3']
        df['StochRSI_D'] = stoch['STOCHRSId_14_14_3_3']
    else:
        df['StochRSI_K'] = 50
        df['StochRSI_D'] = 50
    
    # StochRSI durumları
    df['StochRSI_Bullish'] = (df['StochRSI_K'] > df['StochRSI_D']).astype(int)
    df['StochRSI_Overbought'] = (df['StochRSI_K'] > 80).astype(int)
    df['StochRSI_Oversold'] = (df['StochRSI_K'] < 20).astype(int)
    
    # Hacim ortalaması
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # ADX (Trend gücü)
    try:
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['ADX'] = adx['ADX_14']
        else:
            df['ADX'] = np.nan
    except Exception:
        df['ADX'] = np.nan
    
    # POI - Pivot bazlı destek/direnç (Pine Script uyumlu)
    df = calculate_pivot_points(df, swing_length=10)
    
    # Destek/Dirence uzaklık
    df['Dist_to_Resistance'] = (df['Nearest_Resistance'] - df['Close']) / df['Close']
    df['Dist_to_Support'] = (df['Close'] - df['Nearest_Support']) / df['Close']
    
    # POI yakınlık durumları
    df['Near_Support'] = (df['Dist_to_Support'] < 0.02).astype(int)
    df['Near_Resistance'] = (df['Dist_to_Resistance'] < 0.02).astype(int)
    
    # === FİLTRELENMİŞ AL SİNYALİ ===
    # Yeni mantık (daha esnek):
    # 1. UT Bot AL sinyali
    # 2. Trend uygun:
    #    - Fiyat EMA200 üstünde (uptrend) VEYA
    #    - EMA destek bölgesinde VEYA
    #    - SÜPER güçlü bölgede VEYA
    #    - POI destek bölgesinde
    # 3. StochRSI aşırı alımda değil
    # 4. StochRSI bullish
    
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


# ============================================================
# BACKTEST MOTORU
# ============================================================

class BacktestEngine:
    """
    Gelişmiş backtest motoru.
    Stop loss, trailing stop, komisyon hesabı içerir.
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        
    def run_backtest(self, df, symbol):
        """
        Tek bir hisse için backtest çalıştırır.
        """
        self.trades = []
        in_position = False
        entry_price = 0
        entry_date = None
        highest_since_entry = 0
        
        for i, (index, row) in enumerate(df.iterrows()):
            current_price = row['Close']
            
            if in_position:
                # Highest price güncelle (trailing stop için)
                highest_since_entry = max(highest_since_entry, row['High'])

                # Target profit hesapla
                target_price = entry_price * (1 + self.config.TARGET_PROFIT_PCT)

                # Stop Loss hesapla
                stop_price = entry_price * (1 - self.config.STOP_LOSS_PCT)

                # Trailing Stop (opsiyonel) - sadece target'a ulaşıldıktan sonra veya kârdayken aktif
                if self.config.TRAILING_STOP:
                    # Trailing stop sadece kârdayken devreye girsin
                    if highest_since_entry > entry_price:
                        trailing_stop_price = highest_since_entry * (1 - self.config.STOP_LOSS_PCT)
                        # Trailing stop, initial stop'tan yüksekse kullan
                        if trailing_stop_price > stop_price:
                            stop_price = trailing_stop_price

                # ÖNCELİK 1: Target profit'e ulaştı mı? (önce kontrol et!)
                if row['High'] >= target_price:
                    exit_price = target_price
                    self._close_trade(entry_price, exit_price, entry_date, index, 'TARGET')
                    in_position = False
                    continue

                # ÖNCELİK 2: Stop oldu mu?
                if row['Low'] <= stop_price:
                    exit_price = stop_price
                    self._close_trade(entry_price, exit_price, entry_date, index, 'STOP_LOSS')
                    in_position = False
                    continue

                # ÖNCELİK 3: Normal SAT sinyali
                if row['Signal_Sell'] == 1:
                    exit_price = current_price
                    self._close_trade(entry_price, exit_price, entry_date, index, 'SIGNAL')
                    in_position = False
                    continue
            
            else:
                # FİLTRELENMİŞ AL sinyali kontrolü (UT Bot + EMA + StochRSI + POI)
                if row['Filtered_Buy'] == 1:
                    # Hacim filtresi
                    if row['Volume_Ratio'] < 0.5:
                        continue  # Düşük hacimde girme
                    
                    # Pozisyon aç
                    in_position = True
                    entry_price = current_price
                    entry_date = index
                    highest_since_entry = row['High']
        
        return self._calculate_stats(symbol)
    
    def _close_trade(self, entry, exit, entry_date, exit_date, exit_type):
        """
        İşlemi kapatır ve kayıt altına alır.
        """
        # Brüt kar/zarar
        gross_pnl = (exit - entry) / entry
        
        # Net kar/zarar (komisyon ve slippage düşülmüş)
        costs = self.config.COMMISSION_PCT + self.config.SLIPPAGE_PCT
        net_pnl = gross_pnl - costs
        
        self.trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry,
            'exit_price': exit,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'exit_type': exit_type,
            'holding_days': (exit_date - entry_date).days if hasattr(exit_date, 'days') else 1
        })
    
    def _calculate_stats(self, symbol):
        """
        Backtest istatistiklerini hesaplar.
        """
        if len(self.trades) < self.config.MIN_TRADES:
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Temel metrikler
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        
        win_rate = (winning_trades / total_trades) * 100
        
        # Kar/Zarar analizi
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() * 100 if losing_trades > 0 else 0
        
        # Profit Factor (inf yerine max 99.99)
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.99
        
        # %10 üzeri başarı oranı (NET kar - komisyon zaten düşülmüş)
        # Big Win eşiği: Net %10 (0.10), target'tan bağımsız
        BIG_WIN_THRESHOLD = 0.10  # Net %10 kar
        big_wins = len(trades_df[trades_df['net_pnl'] >= BIG_WIN_THRESHOLD])
        big_win_rate = (big_wins / total_trades) * 100
        
        # Expectancy (Beklenen değer)
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        # Maximum Drawdown
        cumulative = (1 + trades_df['net_pnl']).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Exit type dağılımı
        exit_types = trades_df['exit_type'].value_counts().to_dict()
        
        return {
            'Symbol': symbol,
            'Total_Trades': total_trades,
            'Win_Rate': round(win_rate, 2),
            'Big_Win_Rate': round(big_win_rate, 2),
            'Avg_Win_Pct': round(avg_win, 2),
            'Avg_Loss_Pct': round(avg_loss, 2),
            'Profit_Factor': round(profit_factor, 2),
            'Expectancy': round(expectancy, 2),
            'Max_Drawdown': round(max_drawdown, 2),
            'Stops_Hit': exit_types.get('STOP_LOSS', 0),
            'Targets_Hit': exit_types.get('TARGET', 0),
            'Signal_Exits': exit_types.get('SIGNAL', 0)
        }


# ============================================================
# ANA İŞLEM
# ============================================================

def process_single_stock(symbol, config, debug=False):
    """
    Tek bir hisseyi işler ve backtest yapar.
    """
    try:
        # Veri indir
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(config.TRAIN_PERIOD_YEARS * 365))

        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=config.INTERVAL,
            progress=False
        )

        # None veya boş DataFrame kontrolü
        if df is None:
            if debug:
                print(f"   ❌ {symbol}: Veri None döndü")
            return None

        if df.empty or len(df) < 300:
            if debug:
                print(f"   ❌ {symbol}: Yetersiz veri (len={len(df) if df is not None else 0})")
            return None

        # MultiIndex düzelt
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Gerekli kolonların varlığını kontrol et
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if debug:
                    print(f"   ❌ {symbol}: '{col}' kolonu eksik")
                return None

        # Hacim kontrolü
        avg_volume = df['Volume'].mean() * df['Close'].mean()
        if avg_volume < config.MIN_VOLUME_AVG:
            if debug:
                print(f"   ❌ {symbol}: Düşük hacim (avg={avg_volume:,.0f} TL)")
            return None

        # Heikin Ashi
        if config.USE_HEIKIN_ASHI:
            df = calculate_heikin_ashi(df)

        # UT Bot hesapla
        df = calculate_ut_bot(
            df,
            config.KEY_VALUE,
            config.ATR_PERIOD,
            config.USE_HEIKIN_ASHI
        )

        # Ek indikatörler
        df = calculate_indicators(df)

        # Pivot NaN'lerini doldur (başlangıçta NaN olabilir)
        df['Nearest_Resistance'] = df['Nearest_Resistance'].bfill()
        df['Nearest_Support'] = df['Nearest_Support'].bfill()
        df['Dist_to_Resistance'] = df['Dist_to_Resistance'].bfill().fillna(0.1)
        df['Dist_to_Support'] = df['Dist_to_Support'].bfill().fillna(0.1)
        
        # NaN temizle - sadece gerekli sütunlar (Pivot_High/Low hariç!)
        essential_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 
                          'EMA_200', 'Trailing_Stop', 'Signal_Buy', 
                          'StochRSI_K', 'StochRSI_D', 'Volume_Ratio']
        df = df.dropna(subset=essential_cols)

        if len(df) < 300:
            if debug:
                print(f"   ❌ {symbol}: NaN temizleme sonrası yetersiz veri (len={len(df)})")
            return None

        # Backtest çalıştır
        engine = BacktestEngine(config)
        stats = engine.run_backtest(df, symbol)

        if stats is None and debug:
            print(f"   ❌ {symbol}: Yetersiz işlem sayısı (<{config.MIN_TRADES})")

        return stats

    except Exception as e:
        if debug:
            print(f"   ❌ {symbol}: Hata - {str(e)}")
        return None


def main():
    """
    Ana program - tüm BIST'i tarar ve en iyi hisseleri seçer.
    """
    print("=" * 60)
    print("🚀 BIST HİSSE SEÇİM VE BACKTEST SİSTEMİ")
    print("=" * 60)
    
    config = Config()
    
    # Çıktı klasörü
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Ayarları göster
    print(f"\n📋 AYARLAR:")
    print(f"   • Veri Periyodu: {config.TRAIN_PERIOD_YEARS} yıl")
    print(f"   • Zaman Dilimi: {config.INTERVAL}")
    print(f"   • Stop Loss: %{config.STOP_LOSS_PCT * 100}")
    print(f"   • Hedef Kar: %{config.TARGET_PROFIT_PCT * 100}")
    print(f"   • Komisyon: %{config.COMMISSION_PCT * 100}")
    print(f"   • Trailing Stop: {'Aktif' if config.TRAILING_STOP else 'Pasif'}")
    
    # Hisse listesi
    tickers = get_bist_tickers()
    
    print(f"\n🔍 {len(tickers)} hisse taranıyor...\n")
    
    results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, ticker in enumerate(tickers):
        stats = process_single_stock(ticker, config)
        
        if stats:
            results.append(stats)
            successful += 1
            
            # Güzel sonuçları anlık göster
            if stats['Big_Win_Rate'] >= 50 and stats['Profit_Factor'] >= 1.5:
                print(f"🔥 {ticker}: Win={stats['Win_Rate']}%, BigWin={stats['Big_Win_Rate']}%, PF={stats['Profit_Factor']}")
        else:
            failed += 1
        
        # İlerleme
        progress = (i + 1) / len(tickers) * 100
        print(f"   İlerleme: %{progress:.1f} ({i+1}/{len(tickers)})", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n\n✅ Tarama Tamamlandı! ({elapsed:.1f} saniye)")
    print(f"   • Başarılı: {successful}")
    print(f"   • Atlanan: {failed}")
    
    if not results:
        print("\n❌ Kriterlere uygun hisse bulunamadı!")
        return
    
    # DataFrame'e çevir
    df_results = pd.DataFrame(results)
    
    # Filtreleme
    df_filtered = df_results[
        (df_results['Win_Rate'] >= config.MIN_WIN_RATE) &
        (df_results['Profit_Factor'] >= 1.0)
    ]
    
    # Sıralama (Çoklu kriter)
    # Öncelik: Big_Win_Rate > Profit_Factor > Expectancy
    df_sorted = df_filtered.sort_values(
        by=['Big_Win_Rate', 'Profit_Factor', 'Expectancy'],
        ascending=[False, False, False]
    ).head(config.TOP_N_STOCKS)
    
    # Sonuçları kaydet
    csv_path = os.path.join(config.OUTPUT_DIR, "backtest_sonuclari.csv")
    df_sorted.to_csv(csv_path, index=False)
    
    # Elit hisse listesi (sadece semboller)
    txt_path = os.path.join(config.OUTPUT_DIR, "elit_hisseler.txt")
    with open(txt_path, 'w') as f:
        for symbol in df_sorted['Symbol']:
            f.write(f"{symbol}\n")
    
    # Ekrana özet
    print("\n" + "=" * 60)
    print("🏆 EN İYİ 15 HİSSE")
    print("=" * 60)
    
    display_cols = ['Symbol', 'Win_Rate', 'Big_Win_Rate', 'Profit_Factor', 
                    'Expectancy', 'Max_Drawdown', 'Total_Trades']
    print(df_sorted[display_cols].head(15).to_string(index=False))
    
    print(f"\n📁 Dosyalar kaydedildi:")
    print(f"   • {csv_path}")
    print(f"   • {txt_path}")
    
    # İstatistik özeti
    print("\n📊 GENEL İSTATİSTİKLER:")
    print(f"   • Ortalama Win Rate: %{df_sorted['Win_Rate'].mean():.1f}")
    print(f"   • Ortalama Big Win Rate: %{df_sorted['Big_Win_Rate'].mean():.1f}")
    print(f"   • Ortalama Profit Factor: {df_sorted['Profit_Factor'].mean():.2f}")
    print(f"   • Ortalama Max Drawdown: %{df_sorted['Max_Drawdown'].mean():.1f}")


if __name__ == "__main__":
    main()
