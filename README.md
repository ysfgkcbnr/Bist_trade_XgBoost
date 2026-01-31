# BIST Trade XGBoost

BIST (Borsa Istanbul) hisseleri icin XGBoost tabanli makine ogrenmesi sinyal filtreleme sistemi.

## Sistem Nasil Calisiyor?

```
TradingView (UT Bot Sinyali) --> Webhook Server --> XGBoost Model --> Telegram Bildirimi
```

1. **TradingView** uzerinde UT Bot indikatoru AL sinyali uretir
2. Sinyal **webhook** uzerinden sunucuya gonderilir
3. **XGBoost modeli** sinyali degerlendirir (gec/gecme karari)
4. Onaylanan sinyaller **Telegram**'a bildirim olarak gonderilir

## Dosyalar

| Dosya | Aciklama |
|-------|----------|
| `model_egitimi.py` | XGBoost modellerini egitir (hisse bazli) |
| `webhook_server.py` | Flask webhook sunucusu - sinyalleri alir ve filtreler |
| `TRADE_to_WIN_WEBHOOK.pine` | TradingView Pine Script indikatoru |
| `models/` | Egitilmis XGBoost modelleri (.pkl dosyalari) |
| `hisseler.txt` | Tum BIST hisse listesi |
| `egitilmis_hisseler.txt` | Model egitilmis hisseler |

## Ozellikler

### Model Egitimi (`model_egitimi.py`)
- 10 yillik veri ile egitim
- Walk-forward validation (Train/Val/Test split)
- Purging ile data leakage onleme
- Early stopping ile overfitting onleme
- TimeSeriesSplit cross-validation
- Hisse bazli ayri model egitimi
- Minimum %65 precision threshold

### Webhook Server (`webhook_server.py`)
- Flask tabanli REST API
- Hisse bazli model yukleme
- BIST100 endeks rejimi kontrolu
- Telegram entegrasyonu
- Guvenlik icin secret key dogrulama

### Pine Script (`TRADE_to_WIN_WEBHOOK.pine`)
- UT Bot (ATR Trailing Stop) sinyalleri
- Supply/Demand zone tespiti
- EMA (200, 377, 610) analizi
- StochRSI gostergesi
- Webhook JSON mesaj formatlama

## Kurulum

### 1. Yerel Kurulum
```bash
# Repoyu klonla
git clone https://github.com/ysfgkcbnr/Bist_trade_XgBoost.git
cd Bist_trade_XgBoost

# Bagimliliklari yukle
pip install -r requirements.txt

# .env dosyasi olustur
cp .env.example .env
# .env dosyasini duzenle ve Telegram bilgilerini gir

# Sunucuyu calistir
python webhook_server.py
```

### 2. Render.com Deployment
1. GitHub reposunu Render'a bagla
2. Environment variables ekle:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `WEBHOOK_SECRET`
3. Deploy et

### 3. TradingView Ayarlari
1. Pine Script'i TradingView'e ekle
2. Alert olustur:
   - Condition: `TRADE_to_WIN-WEBHOOK` > `Any alert()`
   - Webhook URL: `https://your-app.onrender.com/webhook`
   - Message: `{{message}}`

## API Endpoints

| Endpoint | Method | Aciklama |
|----------|--------|----------|
| `/webhook` | POST | TradingView sinyallerini alir |
| `/health` | GET | Sistem durumu kontrolu |
| `/test` | GET | Telegram baglanti testi |
| `/stats` | GET | Model istatistikleri |

## Model Metrikleri

- **Precision**: Modelin "AL" dedigi sinyallerin kac tanesi gercekten kar etti
- **Accuracy**: Genel dogru tahmin orani
- **Threshold**: Minimum %65 precision

## Egitilmis Hisseler

Sistem su anda 42 hisse icin model iceriyor. Tam liste `egitilmis_hisseler.txt` dosyasinda.

## Teknolojiler

- Python 3.11+
- XGBoost
- Flask + Gunicorn
- Pandas, NumPy
- yfinance (veri cekme)
- TradingView Pine Script v6

## Lisans

MIT License

## Uyari

Bu sistem yatirim tavsiyesi degildir. Finansal kararlarinizi kendi arastirmaniza dayandirin.
