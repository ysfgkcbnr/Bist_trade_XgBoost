"""
WEBHOOK SUNUCUSU - CANLI TİCARET SİSTEMİ
=========================================
TradingView'den gelen sinyalleri alır,
ML modeline sorar ve Telegram'a bildirim gönderir.

Kullanım:
1. Bu dosyayı sunucunda çalıştır
2. TradingView'de alert webhook URL'ini ayarla
3. Telegram bot token ve chat ID'ni gir

Yazar: Trading AI System
Versiyon: 2.0
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
from datetime import datetime, timedelta
import logging
import yfinance as yf

# .env dosyasını yükle (lokal geliştirme için)
# Railway'de bu dosya olmayacak, environment variables kullanılacak
from pathlib import Path
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webhook.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# AYARLAR - BUNLARI KENDİ BİLGİLERİNLE DEĞİŞTİR!
# ============================================================

class WebhookConfig:
    # Telegram Ayarları (Environment Variable'dan alınır - Railway'de ayarla)
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

    # Model Yolları (HİSSE BAZLI)
    MODELS_DIR = "models"                       # Her hisse için ayrı model
    FEATURES_PATH = "models/feature_columns.json"
    SUMMARY_PATH = "models/model_summary.json"

    # Sinyal Eşik Değeri
    CONFIDENCE_THRESHOLD = 0.65  # %65 güven altındaki sinyalleri gönderme

    # Güvenlik (Environment Variable'dan alınır)
    WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', 'default_secret')

    # Endeks Ayarları
    INDEX_SYMBOL = "XU100.IS"                   # BIST100 endeksi
    INDEX_UPDATE_INTERVAL_MIN = 60              # Endeks verisi güncelleme sıklığı (dakika)


# ============================================================
# ÖZEL MODELLER (Deserialize için gerekli)
# ============================================================

class DominantWinnerModel:
    """
    %100 başarı oranına sahip hisseler için dummy model.
    model_egitimi.py ile aynı yapıda olması gerekir.
    """
    def __init__(self, confidence=0.85):
        self.is_dominant = True
        self.confidence = confidence

    def predict(self, X):
        return np.ones(len(X))

    def predict_proba(self, X):
        return np.column_stack([
            np.full(len(X), 1 - self.confidence),
            np.full(len(X), self.confidence)
        ])


# Model cache ve feature'ları global olarak yükle
MODEL_CACHE = {}  # {"THYAO": model, "GARAN": model, ...}
FEATURE_COLS = None
MODEL_SUMMARY = None
AVAILABLE_STOCKS = []  # Eğitilmiş hisse listesi


def load_features_and_summary():
    """Feature listesi ve model özetini yükler."""
    global FEATURE_COLS, MODEL_SUMMARY, AVAILABLE_STOCKS
    
    config = WebhookConfig()
    
    try:
        # Feature columns
        if os.path.exists(config.FEATURES_PATH):
            with open(config.FEATURES_PATH, 'r') as f:
                FEATURE_COLS = json.load(f)
            logger.info(f"✅ {len(FEATURE_COLS)} feature yüklendi")
        else:
            logger.error(f"❌ Feature listesi bulunamadı: {config.FEATURES_PATH}")
            return False
        
        # Model summary
        if os.path.exists(config.SUMMARY_PATH):
            with open(config.SUMMARY_PATH, 'r', encoding='utf-8') as f:
                MODEL_SUMMARY = json.load(f)
            AVAILABLE_STOCKS = [m['symbol_clean'] for m in MODEL_SUMMARY.get('models', [])]
            # Liste çok uzun olabildiği için ilk 10 ve son 5'i göster
            stock_sample = f"{AVAILABLE_STOCKS[:10]} ... {AVAILABLE_STOCKS[-5:]}" if len(AVAILABLE_STOCKS) > 15 else AVAILABLE_STOCKS
            logger.info(f"✅ {len(AVAILABLE_STOCKS)} hisse için Elit Model mevcut: {stock_sample}")
        else:
            logger.warning(f"⚠️ Model özeti bulunamadı: {config.SUMMARY_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Yükleme hatası: {e}")
        return False


def load_stock_model(symbol_clean):
    """
    Belirli bir hisse için modeli yükler (önbellek kullanır).
    
    Args:
        symbol_clean: Hisse sembolü (orn: "THYAO", ".IS" olmadan)
    
    Returns:
        Model veya None
    """
    global MODEL_CACHE
    
    # Önbellekte varsa döndür
    if symbol_clean in MODEL_CACHE:
        return MODEL_CACHE[symbol_clean]
    
    # Diskten yükle
    config = WebhookConfig()
    model_path = os.path.join(config.MODELS_DIR, f"{symbol_clean}.pkl")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            MODEL_CACHE[symbol_clean] = model  # Önbelleğe ekle
            logger.info(f"✅ Model yüklendi: {symbol_clean}")
            return model
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası ({symbol_clean}): {e}")
            return None
    else:
        logger.warning(f"⚠️ Model bulunamadı: {model_path}")
        return None


def send_telegram_message(message, parse_mode='HTML'):
    """Telegram'a mesaj gönderir."""
    config = WebhookConfig()

    if not config.TELEGRAM_BOT_TOKEN:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN environment variable ayarlanmamış!")
        return False
    
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    
    payload = {
        'chat_id': config.TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': parse_mode
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("📤 Telegram mesajı gönderildi")
            return True
        else:
            logger.error(f"Telegram hatası: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Telegram bağlantı hatası: {e}")
        return False


# Endeks veri önbelleği
INDEX_CACHE = {
    'data': None,
    'last_update': datetime.min,
    'features': {
        'Index_Trend': 1,
        'Rel_Strength_10': 0.0,
        'Rel_Strength_30': 0.0,
        'Index_ADX': 25.0
    }
}

def get_index_features(symbol_clean, price):
    """BIST100 verilerini alır ve hisse için relatif özellikleri hesaplar."""
    global INDEX_CACHE
    config = WebhookConfig()
    now = datetime.now()
    
    # Güncelleme zamanı kontrolü
    if now - INDEX_CACHE['last_update'] > timedelta(minutes=config.INDEX_UPDATE_INTERVAL_MIN):
        try:
            logger.info("📥 BIST100 verisi güncelleniyor...")
            idx = yf.download(config.INDEX_SYMBOL, period="1y", interval="1d", progress=False)
            if not idx.empty:
                if isinstance(idx.columns, pd.MultiIndex):
                    idx.columns = idx.columns.get_level_values(0)
                
                # Endeks Trendi
                ema200 = idx['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                idx_close = idx['Close'].iloc[-1]
                INDEX_CACHE['features']['Index_Trend'] = 1 if idx_close > ema200 else 0
                
                # Endeks Getirileri (Relatif güç için)
                INDEX_CACHE['data'] = idx
                INDEX_CACHE['last_update'] = now
                logger.info(f"✅ BIST100 güncellendi. Rejim: {'BOĞA' if idx_close > ema200 else 'AYI'}")
        except Exception as e:
            logger.error(f"Endeks verisi alınamadı: {e}")

    # Hisseye özel relatif özellikler
    features = INDEX_CACHE['features'].copy()
    if INDEX_CACHE['data'] is not None:
        try:
            idx = INDEX_CACHE['data']
            # Basit Yaklaşım: Webhook'tan gelen fiyat ile endeks değişimini karşılaştır
            # (Hisse verisi tam olmadığı için dünkü kapanışa göre tahmini güç)
            idx_ret_10 = idx['Close'].pct_change(10).iloc[-1]
            idx_ret_30 = idx['Close'].pct_change(30).iloc[-1]
            
            # Not: Tam relatif güç için hissenin geçmişi lazım, 
            # ancak canlı sinyal anında '0' veya 'pozitif/negatif' tahmini olarak 0.0 bırakıyoruz.
            # Alternatif olarak 0.0 verip modeli Index_Trend'e odaklandırabiliriz.
            features['Rel_Strength_10'] = 0.0
            features['Rel_Strength_30'] = 0.0
            features['Index_ADX'] = 25.0
        except: pass
        
    return features


def prepare_features_from_payload(data):
    """
    TradingView'den gelen veriyi model formatına çevirir.
    
    TÜM 37 FEATURE (Market Regime eklendi):
    """
    try:
        ticker = data.get('ticker', '')
        symbol_clean = ticker.replace('.IS', '').upper()
        price = float(data.get('price', 0))
        
        # Endeks özelliklerini al
        idx_features = get_index_features(symbol_clean, price)

        # Webhook'tan gelen -> Model'in beklediği
        feature_mapping = {
            # EMA Uzaklıkları
            'Dist_EMA_200': data.get('dist_ema_200', 0),
            'Dist_EMA_377': data.get('dist_ema_377', 0),
            'Dist_EMA_610': data.get('dist_ema_610', 0),
            
            # EMA Gap'ler (yakınsama analizi)
            'EMA_200_377_Gap': data.get('ema_200_377_gap', 0.05),
            'EMA_377_610_Gap': data.get('ema_377_610_gap', 0.05),
            'EMA_All_Gap': data.get('ema_all_gap', 0.10),
            'EMAs_Converging': data.get('emas_converging', 0),
            
            # EMA Destek/Direnç
            'EMA_200_Support': data.get('ema_200_support', 0),
            'EMA_200_Resistance': data.get('ema_200_resistance', 0),
            'EMA_377_Support': data.get('ema_377_support', 0),
            'EMA_377_Resistance': data.get('ema_377_resistance', 0),
            'EMA_610_Support': data.get('ema_610_support', 0),
            'EMA_610_Resistance': data.get('ema_610_resistance', 0),
            
            # Destek/Direnç Gücü
            'Support_Strength': data.get('support_strength', 0),
            'Resistance_Strength': data.get('resistance_strength', 0),
            'Near_EMA_Support': data.get('near_ema_support', 0),
            'Near_EMA_Resistance': data.get('near_ema_resistance', 0),
            'Super_Support_Zone': data.get('super_support_zone', 0),
            
            # EMA Durumları (eski)
            'Above_EMA_200': data.get('above_ema_200', 0),
            'Above_EMA_377': data.get('above_ema_377', 0),
            'Above_EMA_610': data.get('above_ema_610', 0),
            'Above_All_EMAs': data.get('above_all_emas', 0),
            'EMA_Stack': data.get('ema_stack', 0),
            
            # POI - Destek/Direnç
            'Dist_to_Resistance': data.get('dist_to_resistance', 0),
            'Dist_to_Support': data.get('dist_to_support', 0),
            'Near_Support': data.get('near_support', 0),
            'Near_Resistance': data.get('near_resistance', 0),
            
            # StochRSI
            'StochRSI_K': data.get('stoch_k', 50),
            'StochRSI_D': data.get('stoch_d', 50),
            'StochRSI_Oversold': 1 if data.get('stoch_k', 50) < 20 else 0,
            'StochRSI_Overbought': 1 if data.get('stoch_k', 50) > 80 else 0,
            'StochRSI_Bullish': 1 if data.get('stoch_k', 50) > data.get('stoch_d', 50) else 0,
            
            # Trailing Stop
            'Dist_to_Stop': data.get('dist_to_stop', 0),

            # Pazar Rejimi (YENİ)
            'Index_Trend': idx_features['Index_Trend'],
            'Rel_Strength_10': idx_features['Rel_Strength_10'],
            'Rel_Strength_30': idx_features['Rel_Strength_30'],
            'Index_ADX': idx_features['Index_ADX'],
        }
        
        # Model feature'larıyla eşleştir
        features = {}
        for col in FEATURE_COLS:
            if col in feature_mapping:
                features[col] = float(feature_mapping[col])
            else:
                features[col] = 0.0
                logger.warning(f"⚠️ Feature bulunamadı: {col}")
        
        # DataFrame oluştur
        df = pd.DataFrame([features])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Feature hazırlama hatası: {e}")
        return None


# ============================================================
# WEBHOOK ENDPOINT'LERİ
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Sunucu sağlık kontrolü."""
    return jsonify({
        'status': 'ok',
        'available_stocks': len(AVAILABLE_STOCKS),
        'features_loaded': FEATURE_COLS is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """
    TradingView'den gelen ana webhook endpoint'i.
    HİSSE BAZLI MODEL KULLANIMI
    """
    config = WebhookConfig()
    
    try:
        # JSON verisini al
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        logger.info(f"📥 Webhook alındı: {data.get('ticker', 'Unknown')}")
        
        # Güvenlik kontrolü (opsiyonel)
        if 'secret' in data:
            if data['secret'] != config.WEBHOOK_SECRET:
                logger.warning("⚠️ Geçersiz secret key!")
                return jsonify({'error': 'Invalid secret'}), 401
        
        # Gerekli alanları kontrol et
        if 'ticker' not in data:
            return jsonify({'error': 'Missing ticker'}), 400
        
        ticker = data['ticker']
        price = data.get('price', 0)
        signal_type = data.get('signal', 'BUY')
        
        # Ticker'dan sembolü çıkar (THYAO.IS -> THYAO)
        symbol_clean = ticker.replace('.IS', '').upper()
        
        # Bu hisse için model var mı?
        if symbol_clean not in AVAILABLE_STOCKS:
            logger.warning(f"⚠️ {symbol_clean} için model yok, sinyal gönderilmiyor")
            return jsonify({
                'status': 'no_model',
                'ticker': ticker,
                'reason': f'{symbol_clean} için eğitilmiş model bulunamadı'
            }), 200
        
        # Hisseye özel modeli yükle
        model = load_stock_model(symbol_clean)
        
        if model is None:
            logger.error(f"Model yüklenemedi: {symbol_clean}")
            # Model yoksa uyarı ile gönder
            message = f"""
🚨 <b>AL SİNYALİ</b> (Model Yüklenemedi)

📈 <b>{ticker}</b>
💰 Fiyat: {price}
⏰ {datetime.now().strftime('%H:%M:%S')}

⚠️ <i>AI filtreleme başarısız - dikkatli ol!</i>
"""
            send_telegram_message(message)
            return jsonify({'status': 'sent_without_ai'}), 200
        
        # Feature'ları hazırla
        features_df = prepare_features_from_payload(data)
        
        if features_df is None:
            logger.error("Feature hazırlanamadı!")
            return jsonify({'error': 'Feature preparation failed'}), 500
        
        # Model tahmini al (hisseye özel model ile)
        try:
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            confidence = probability[1]  # Sınıf 1 (AL) olasılığı
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return jsonify({'error': 'Prediction failed'}), 500
        
        logger.info(f"🤖 {symbol_clean} Model sonucu: {'AL' if prediction == 1 else 'BEKLEME'} (Güven: %{confidence*100:.1f})")
        
        # Eşik kontrolü
        if confidence >= config.CONFIDENCE_THRESHOLD and prediction == 1:
            # 🟢 GÜÇLÜ SİNYAL - Telegram'a gönder
            
            # Güven seviyesine göre emoji
            if confidence >= 0.85:
                strength = "🔥🔥🔥 ÇOOK GÜÇLÜ"
            elif confidence >= 0.75:
                strength = "🔥🔥 GÜÇLÜ"
            else:
                strength = "🔥 ORTA"
            
            message = f"""
🚀 <b>AL SİNYALİ - AI ONAYLADI</b>

📈 <b>{ticker}</b>
💰 Fiyat: {price}
🎯 AI Güven: <b>%{confidence*100:.1f}</b>
💪 Güç: {strength}

📊 Detaylar:
• RSI: {data.get('rsi', 'N/A')}
• Hacim: {data.get('volume_ratio', 'N/A')}x
• ADX: {data.get('adx', 'N/A')}

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

⚠️ <i>Stop Loss: Girişin %5 altı</i>
"""
            send_telegram_message(message)
            
            return jsonify({
                'status': 'signal_sent',
                'ticker': ticker,
                'confidence': confidence,
                'recommendation': 'BUY'
            }), 200
        
        else:
            # 🔴 ZAYIF SİNYAL - Gönderme
            logger.info(f"❌ {ticker} sinyali elendi (Güven: %{confidence*100:.1f})")
            
            return jsonify({
                'status': 'signal_rejected',
                'ticker': ticker,
                'confidence': confidence,
                'reason': 'Below threshold' if confidence < config.CONFIDENCE_THRESHOLD else 'Model says wait'
            }), 200
            
    except Exception as e:
        logger.error(f"Webhook hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['GET'])
def test_telegram():
    """Telegram bağlantısını test eder."""
    success = send_telegram_message("🔔 <b>Test Mesajı</b>\n\nWebhook sunucusu çalışıyor!")
    
    if success:
        return jsonify({'status': 'Telegram OK'}), 200
    else:
        return jsonify({'error': 'Telegram failed'}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Hisse bazlı model istatistiklerini döndürür."""
    if MODEL_SUMMARY:
        return jsonify({
            'system': 'HİSSE BAZLI MODEL',
            'training_date': MODEL_SUMMARY.get('training_date', 'Unknown'),
            'total_models': MODEL_SUMMARY.get('trained_count', 0),
            'available_stocks': AVAILABLE_STOCKS,
            'cached_models': list(MODEL_CACHE.keys()),
            'features_count': len(FEATURE_COLS) if FEATURE_COLS else 0
        })
    else:
        return jsonify({'error': 'Model özeti bulunamadı'}), 404


# ============================================================
# PINE SCRIPT ÖRNEK KODU (TradingView için)
# ============================================================

PINE_SCRIPT_TEMPLATE = """
// ============================================================
// TradingView Webhook - UT Bot + ML Sistemi
// SADECE SENİN STRATEJİN: EMA, POI, StochRSI
// ============================================================

// === EMA'LAR (200, 377, 610) ===
ema200 = ta.ema(close, 200)
ema377 = ta.ema(close, 377)
ema610 = ta.ema(close, 610)

// === STOCHASTIC RSI (K=3, D=3, RSI=14, Stoch=14) ===
smoothK = 3
smoothD = 3
lengthRSI = 14
lengthStoch = 14
rsi1 = ta.rsi(close, lengthRSI)
stoch_k = ta.sma(ta.stoch(rsi1, rsi1, rsi1, lengthStoch), smoothK)
stoch_d = ta.sma(stoch_k, smoothD)

// === POI - DESTEK/DİRENÇ (swing_length = 10) ===
swing_length = 10
swing_high = ta.highest(high, swing_length)
swing_low = ta.lowest(low, swing_length)
dist_to_res = (swing_high - close) / close
dist_to_sup = (close - swing_low) / close

// === TRAILING STOP UZAKLIĞI ===
// xATRTrailingStop senin indikatöründe zaten var
dist_to_stop = (close - xATRTrailingStop) / close

// === WEBHOOK JSON ===
alert_message = '{' +
    '"ticker": "' + syminfo.ticker + '",' +
    '"price": ' + str.tostring(close) + ',' +
    '"signal": "BUY",' +
    
    // EMA Uzaklıkları
    '"dist_ema_200": ' + str.tostring((close - ema200) / ema200) + ',' +
    '"dist_ema_377": ' + str.tostring((close - ema377) / ema377) + ',' +
    '"dist_ema_610": ' + str.tostring((close - ema610) / ema610) + ',' +
    
    // EMA Durumları
    '"above_ema_200": ' + str.tostring(close > ema200 ? 1 : 0) + ',' +
    '"above_ema_377": ' + str.tostring(close > ema377 ? 1 : 0) + ',' +
    '"above_ema_610": ' + str.tostring(close > ema610 ? 1 : 0) + ',' +
    '"above_all_emas": ' + str.tostring(close > ema200 and close > ema377 and close > ema610 ? 1 : 0) + ',' +
    '"ema_stack": ' + str.tostring(ema200 > ema377 and ema377 > ema610 ? 1 : 0) + ',' +
    
    // POI - Destek/Direnç
    '"dist_to_resistance": ' + str.tostring(dist_to_res) + ',' +
    '"dist_to_support": ' + str.tostring(dist_to_sup) + ',' +
    '"near_support": ' + str.tostring(dist_to_sup < 0.02 ? 1 : 0) + ',' +
    '"near_resistance": ' + str.tostring(dist_to_res < 0.02 ? 1 : 0) + ',' +
    
    // StochRSI
    '"stoch_k": ' + str.tostring(stoch_k) + ',' +
    '"stoch_d": ' + str.tostring(stoch_d) + ',' +
    
    // Trailing Stop
    '"dist_to_stop": ' + str.tostring(dist_to_stop) + ',' +
    
    // Güvenlik
    '"secret": "your_secret_key_here"' +
'}'

// === ALERT - Senin AL sinyalin ===
if buy  // buy = src > xATRTrailingStop and above
    alert(alert_message, alert.freq_once_per_bar_close)
"""


@app.route('/pinescript', methods=['GET'])
def get_pinescript():
    """Pine Script örnek kodunu döndürür."""
    return PINE_SCRIPT_TEMPLATE, 200, {'Content-Type': 'text/plain'}


# ============================================================
# BAŞLATMA
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 WEBHOOK SUNUCUSU BAŞLATILIYOR (HİSSE BAZLI MODEL)")
    print("=" * 60)
    
    # Feature ve model özetini yükle
    if load_features_and_summary():
        print("\n✅ Sistem hazır!")
        print(f"   • Mevcut hisseler: {len(AVAILABLE_STOCKS)}")
        print(f"   • Hisse listesi: {', '.join(AVAILABLE_STOCKS[:5])}{'...' if len(AVAILABLE_STOCKS) > 5 else ''}")
        print(f"   • Feature sayısı: {len(FEATURE_COLS) if FEATURE_COLS else 'N/A'}")
    else:
        print("\n⚠️ Model özeti yüklenemedi - Önce model eğitimi yapın!")
    
    print("\n📡 Endpoint'ler:")
    print("   • POST /webhook - TradingView sinyalleri")
    print("   • GET  /health  - Sağlık kontrolü")
    print("   • GET  /test    - Telegram testi")
    print("   • GET  /stats   - Model istatistikleri")
    print("   • GET  /pinescript - Pine Script örneği")
    
    # Railway veya lokal için port ayarı
    port = int(os.environ.get('PORT', 5001))

    print(f"\n🔗 Sunucu başlatılıyor: http://0.0.0.0:{port}")
    if port == 5001:
        print("   (ngrok kullanıyorsan: ngrok http 5001)")

    # Flask'ı başlat
    app.run(host='0.0.0.0', port=port, debug=False)
