import io, json, pickle, base64, warnings, asyncio, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional
 
import cv2
import numpy as np
import pandas as pd
from PIL import Image
 
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
 
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
 
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
print(f"Loading models on {DEVICE}...")
 
# ── Load configs ──
with open(BASE_DIR / 'class_names.json')   as f: CLASS_NAMES   = json.load(f)
with open(BASE_DIR / 'lstm_config.json')   as f: lstm_cfg       = json.load(f)
with open(BASE_DIR / 'fusion_config.json') as f: fusion_cfg     = json.load(f)
 
NUM_CLASSES      = len(CLASS_NAMES)
WEATHER_FEATURES = lstm_cfg['weather_features']
DISEASE_CLASSES  = lstm_cfg['disease_classes']
LOOKBACK         = lstm_cfg['lookback']
FORECAST         = lstm_cfg['forecast_steps']
N_DISEASES       = lstm_cfg['n_diseases']
N_FEATURES       = len(WEATHER_FEATURES)
CROP_TYPES       = fusion_cfg['crop_types']
GROWTH_STAGES    = fusion_cfg['growth_stages']
META_DIM         = fusion_cfg['meta_dim']
CNN_DIM          = 512
WEATHER_DIM      = 256
FUSION_INPUT_DIM = CNN_DIM + WEATHER_DIM + META_DIM
IMG_SIZE         = 380
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
 
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])
 
# ── CNN ──
class CropDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=38, dropout=0.4):
        super().__init__()
        backbone        = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        self.embedding  = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(1792, 512), nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)
        self._feat_maps  = None
        self._feat_grads = None
        self.features[-1].register_forward_hook(
            lambda m, i, o: setattr(self, '_feat_maps', o))
        self.features[-1].register_full_backward_hook(
            lambda m, i, o: setattr(self, '_feat_grads', o[0]))
 
    def forward(self, x):
        x = self.features(x); x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(self.embedding(x))
 
    def get_embedding(self, x):
        with torch.no_grad():
            x = self.features(x); x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.embedding(x)
 
    def grad_cam(self, tensor, class_idx=None):
        self.eval()
        x      = tensor.unsqueeze(0).to(DEVICE)
        logits = self.forward(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.zero_grad()
        logits[0, class_idx].backward()
        weights = self._feat_grads.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self._feat_maps).sum(dim=1).squeeze()
        cam     = torch.clamp(cam, min=0)
        cam     = cam / (cam.max() + 1e-8)
        conf    = logits.softmax(dim=1)[0, class_idx].item()
        return cam.detach().cpu().numpy(), class_idx, conf
 
 
cnn_model = CropDiseaseClassifier(num_classes=NUM_CLASSES).to(DEVICE)
cnn_model.load_state_dict(
    torch.load(BASE_DIR / 'best_model_v2.pth', map_location=DEVICE))
cnn_model.eval()
for p in cnn_model.parameters(): p.requires_grad = False
print("CNN loaded ✓")
 
# ── LSTM ──
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
    def forward(self, lstm_out):
        weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * lstm_out).sum(dim=1), weights
 
class WeatherLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 forecast_steps=7, n_diseases=26, dropout=0.3):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.n_diseases     = n_diseases
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = TemporalAttention(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, forecast_steps * n_diseases), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, w = self.attention(out)
        ctx    = self.dropout(ctx)
        pred   = self.fc(ctx).view(-1, self.forecast_steps, self.n_diseases)
        return pred, ctx, w
    def get_weather_context(self, x):
        with torch.no_grad():
            out, _ = self.lstm(x)
            ctx, _ = self.attention(out)
            return ctx
 
lstm_model = WeatherLSTM(
    input_dim=N_FEATURES, hidden_dim=lstm_cfg['hidden_dim'],
    num_layers=lstm_cfg['num_layers'], forecast_steps=FORECAST,
    n_diseases=N_DISEASES, dropout=lstm_cfg['dropout']
).to(DEVICE)
lstm_model.load_state_dict(
    torch.load(BASE_DIR / 'best_lstm.pth', map_location=DEVICE))
lstm_model.eval()
for p in lstm_model.parameters(): p.requires_grad = False
print("LSTM loaded ✓")
 
# ── Fusion ──
class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=512, weather_dim=256, meta_dim=21,
                 n_diseases=26, attn_dim=256, n_heads=8, dropout=0.3):
        super().__init__()
        self.cnn_proj     = nn.Linear(cnn_dim, attn_dim)
        self.weather_proj = nn.Linear(weather_dim, attn_dim)
        self.cross_attn   = nn.MultiheadAttention(attn_dim, n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.attn_norm    = nn.LayerNorm(attn_dim)
        self.self_attn    = nn.MultiheadAttention(attn_dim, n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.self_norm    = nn.LayerNorm(attn_dim)
        self.meta_proj    = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.LayerNorm(64), nn.GELU())
        self.fusion_head  = nn.Sequential(
            nn.Linear(attn_dim*2+64, 512), nn.LayerNorm(512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_diseases)
        )
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        cnn_f = x[:, :CNN_DIM]
        wx_f  = x[:, CNN_DIM:CNN_DIM+WEATHER_DIM]
        meta  = x[:, CNN_DIM+WEATHER_DIM:]
        cnn_q = self.cnn_proj(cnn_f).unsqueeze(1)
        wx_k  = self.weather_proj(wx_f).unsqueeze(1)
        co, _ = self.cross_attn(cnn_q, wx_k, wx_k)
        co    = self.attn_norm(co + cnn_q).squeeze(1)
        so, _ = self.self_attn(cnn_q, cnn_q, cnn_q)
        so    = self.self_norm(so + cnn_q).squeeze(1)
        mo    = self.meta_proj(meta)
        return self.fusion_head(self.dropout(
            torch.cat([co, so, mo], dim=1)))
 
    def predict_with_uncertainty(self, x, n_passes=50):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.train()
        with torch.no_grad():
            preds = torch.stack([
                torch.sigmoid(self.forward(x)) for _ in range(n_passes)])
        return preds.mean(dim=0), preds.std(dim=0)
 
 
fusion_model = CrossAttentionFusion(
    cnn_dim=CNN_DIM, weather_dim=WEATHER_DIM, meta_dim=META_DIM,
    n_diseases=N_DISEASES, attn_dim=256, n_heads=8, dropout=0.3
).to(DEVICE)
fusion_model.load_state_dict(
    torch.load(BASE_DIR / 'best_fusion.pth', map_location=DEVICE))
fusion_model.eval()
print("Fusion model loaded ✓")
 
# ── Calibrators, XGBoost, scaler ──
with open(BASE_DIR / 'calibrators.pkl', 'rb') as f:
    calibrators = pickle.load(f)
with open(BASE_DIR / 'weather_scaler.pkl', 'rb') as f:
    weather_scaler = pickle.load(f)
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(BASE_DIR / 'yield_model.json'))
print("Calibrators, scaler, XGBoost loaded ✓")
 
# ── Weather data ──
weather_df = pd.read_csv(BASE_DIR / 'weather_features.csv',
                          index_col=0, parse_dates=True)
dcws_df    = pd.read_csv(BASE_DIR / 'dcws_scores.csv',
                          index_col=0, parse_dates=True)
print(f"Weather data loaded: {len(weather_df)} days ✓")
print("All models ready. Starting API...")

INTERVENTIONS = {
    'Tomato___Late_blight'       : {'fungicide': 'Mancozeb 75% WP @ 2.5 g/L',
                                    'frequency': 'Every 7 days',
                                    'timing'   : 'Apply at first sign or when risk > 60%',
                                    'cost_per_ha': 800, 'efficacy_pct': 70},
    'Potato___Late_blight'       : {'fungicide': 'Metalaxyl + Mancozeb @ 2.5 g/L',
                                    'frequency': 'Every 7–10 days',
                                    'timing'   : 'Preventive when risk > 50%',
                                    'cost_per_ha': 900, 'efficacy_pct': 75},
    'Tomato___Early_blight'      : {'fungicide': 'Chlorothalonil 75% WP @ 2 g/L',
                                    'frequency': 'Every 10 days',
                                    'timing'   : 'Begin when lower leaves show spots',
                                    'cost_per_ha': 600, 'efficacy_pct': 65},
    'Potato___Early_blight'      : {'fungicide': 'Mancozeb 75% WP @ 2 g/L',
                                    'frequency': 'Every 10–14 days',
                                    'timing'   : 'Start at tuber initiation stage',
                                    'cost_per_ha': 550, 'efficacy_pct': 60},
    'Apple___Apple_scab'         : {'fungicide': 'Captan 50% WP @ 2.5 g/L',
                                    'frequency': 'Every 7–10 days during wet weather',
                                    'timing'   : 'Critical: pink bud to petal fall',
                                    'cost_per_ha': 1200, 'efficacy_pct': 70},
    'Tomato___Bacterial_spot'    : {'fungicide': 'Copper oxychloride 50% WP @ 3 g/L',
                                    'frequency': 'Every 7 days',
                                    'timing'   : 'Apply before or at symptom onset',
                                    'cost_per_ha': 700, 'efficacy_pct': 55},
    'Squash___Powdery_mildew'    : {'fungicide': 'Sulphur 80% WP @ 3 g/L',
                                    'frequency': 'Every 10–14 days',
                                    'timing'   : 'At first sign of white powder',
                                    'cost_per_ha': 400, 'efficacy_pct': 75},
    'Corn_(maize)___Northern_Leaf_Blight': {
                                    'fungicide': 'Propiconazole 25% EC @ 1 mL/L',
                                    'frequency': 'Every 14 days',
                                    'timing'   : 'Apply at tasseling if risk is high',
                                    'cost_per_ha': 750, 'efficacy_pct': 65},
}
DEFAULT_INTERVENTION = {'fungicide': 'Consult local agronomist',
                        'frequency': 'Every 7–14 days',
                        'timing'   : 'At first confirmed symptom',
                        'cost_per_ha': 500, 'efficacy_pct': 50}
 
CROP_MAP = {
    'Apple': 'Apple', 'Blueberry': 'Blueberry',
    'Cherry_(including_sour)': 'Cherry', 'Corn_(maize)': 'Corn',
    'Grape': 'Grape', 'Orange': 'Orange', 'Peach': 'Peach',
    'Pepper,_bell': 'Pepper', 'Potato': 'Potato',
    'Raspberry': 'Raspberry', 'Soybean': 'Soybean',
    'Squash': 'Squash', 'Strawberry': 'Strawberry', 'Tomato': 'Tomato',
}
 
 
def encode_metadata(crop_type, growth_stage,
                    days_since_planting, days_to_harvest):
    vec    = np.zeros(META_DIM, dtype=np.float32)
    ct_idx = CROP_TYPES.index(crop_type) if crop_type in CROP_TYPES else 0
    gs_idx = GROWTH_STAGES.index(growth_stage) \
             if growth_stage in GROWTH_STAGES else 2
    vec[ct_idx]                          = 1.0
    vec[len(CROP_TYPES) + gs_idx]        = 1.0
    vec[len(CROP_TYPES)+len(GROWTH_STAGES)]   = min(days_since_planting/180, 1.0)
    vec[len(CROP_TYPES)+len(GROWTH_STAGES)+1] = min(days_to_harvest/90, 1.0)
    return vec
 
 
def numpy_to_b64(img_array: np.ndarray) -> str:
    """Encodes numpy image array to base64 PNG string."""
    _, buf = cv2.imencode('.png', img_array)
    return base64.b64encode(buf).decode('utf-8')
 
 
def run_gradcam(image_array: np.ndarray, class_idx: int) -> str:
    """Returns base64-encoded Grad-CAM overlay."""
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image']
    cam, _, _ = cnn_model.grad_cam(tensor, class_idx)
    cam_up    = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap   = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    overlay   = (0.5 * resized + 0.5 * heatmap).astype(np.uint8)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return numpy_to_b64(overlay_rgb)

# ── Thread pool for blocking ML inference ──
# Keeps the asyncio event loop free; ML work runs in a dedicated OS thread.
_ML_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_worker")
_executor = _ML_POOL

# ── In-memory job queue and tracking store ──
# job_store entry shape:
# {
#   'status': 'pending'|'processing'|'completed'|'failed',
#   'stage' : 'pending'|'cnn'|'lstm'|'fusion'|'done',
#   'result': { ... },
#   'error' : optional str
# }
job_queue = asyncio.Queue(maxsize=64)
job_store = {}

app = FastAPI(
    title       = "Crop Disease Early Warning System API",
    description = "Multimodal AI — CNN + Weather LSTM + Fusion",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

@app.middleware("http")
async def add_ngrok_header(request: Request, call_next):
    """Bypass the ngrok browser warning page for API calls from mobile apps."""
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response
 
 
# ── Request / Response models ──
class AnalyzeRequest(BaseModel):
    lat                 : float = 18.5204
    lon                 : float = 73.8567
    crop_type           : str   = "Tomato"
    growth_stage        : str   = "fruiting"
    days_since_planting : int   = 75
    days_to_harvest     : int   = 20
    area_ha             : float = 1.0
    market_price_per_kg : float = 25.0
    n_mc_passes         : int   = 30
 
 
class ForecastRequest(BaseModel):
    lat      : float = 18.5204
    lon      : float = 73.8567
    top_n    : int   = 10
 
 
# ── Routes ──
 
async def run_cnn(image_array: np.ndarray):
    """Stage 1 - fast CNN classification and embedding."""
    resized = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        logits  = cnn_model(tensor)
        probs   = torch.softmax(logits, dim=1)[0]
        cnn_class = int(probs.argmax())
        cnn_conf  = float(probs[cnn_class])
        top5 = [
            {'class': CLASS_NAMES[i], 'prob': float(probs[i])}
            for i in probs.argsort(descending=True)[:5]
        ]
        embedding = cnn_model.get_embedding(tensor).squeeze(0).cpu().numpy()

    return {
        'cnn_class': cnn_class,
        'cnn_conf': cnn_conf,
        'cnn_top5': top5,
        'cnn_embedding': embedding,
    }


async def run_lstm():
    """Stage 2 - weather LSTM forecast context."""
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)

    return {
        'lstm_forecast': forecast_raw[0].cpu().numpy(),
        'lstm_ctx': wx_ctx.cpu().numpy(),
    }


def run_fusion(cnn_embedding, lstm_ctx, meta_vector, n_mc_passes=30):
    """Stage 3 - fusion inference + calibration + economics."""
    fusion_input = torch.tensor(
        np.concatenate([cnn_embedding.reshape(1, -1), lstm_ctx.reshape(1, -1), meta_vector.reshape(1, -1)], axis=1),
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
        mean_probs, std_probs = fusion_model.predict_with_uncertainty(
            fusion_input, n_passes=n_mc_passes
        )

    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()

    cal_probs = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])

    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])

    return {
        'fusion_raw': mean_probs.tolist(),
        'fusion_uncertainty': std_probs.tolist(),
        'fusion_calibrated': cal_probs.tolist(),
        'top_disease': top_disease,
        'top_risk': top_risk,
        'top_uncertainty': top_unc,
        'top_idx': top_idx,
    }


def assemble_forecast_and_yield(cnn_info, fusion_info, meta):
    """Helper: build final forecast/yield/intervention payload."""
    forecast_raw = fusion_info.get('forecast_raw')
    # Use LSTM forecast from job lsm stage if provided later
    # (In our working model, we can compute from run_lstm stage separately.)

    top_idx = fusion_info['top_idx']
    top_disease = fusion_info['top_disease']
    top_risk = fusion_info['top_risk']

    severity = cnn_info['cnn_conf']
    if CLASS_NAMES[cnn_info['cnn_class']] != top_disease:
        severity = 0.5

    xgb_feat = np.array([[
        top_risk,
        severity,
        GROWTH_STAGES.index(meta['growth_stage']),
        meta['days_to_harvest'],
        0, 0, top_idx
    ]])

    yield_loss = float(xgb_model.predict(xgb_feat)[0])

    rec = INTERVENTIONS.get(top_disease, DEFAULT_INTERVENTION)
    fin_loss = yield_loss * 20000 * meta['area_ha'] * meta['market_price_per_kg']
    treat_cost = rec['cost_per_ha'] * meta['area_ha']
    saved_val = fin_loss * (rec['efficacy_pct'] / 100)

    if top_risk > 0.6:
        urgency = 'TREAT TODAY'
    elif top_risk > 0.3:
        urgency = 'MONITOR — treat within 3 days if worsening'
    else:
        urgency = 'LOW RISK — continue monitoring'

    return {
        'yield': {
            'loss_pct': yield_loss,
            'loss_kg': yield_loss * 20000 * meta['area_ha'],
            'financial_loss': fin_loss,
            'treatment_cost': treat_cost,
            'saved_value': saved_val,
            'roi': (saved_val - treat_cost) / (treat_cost + 1e-8),
        },
        'intervention': {
            'urgency': urgency,
            'fungicide': rec['fungicide'],
            'frequency': rec['frequency'],
            'timing': rec['timing'],
        },
    }


async def process_job(job_id: str):
    """Background job worker executes pipeline stages and updates job_store."""
    job = job_store.get(job_id)
    if job is None:
        return

    try:
        job['status'] = 'processing'
        job['stage']  = 'cnn'

        cnn_info = await asyncio.get_event_loop().run_in_executor(
            _ML_POOL, run_cnn, job['input_image_np']
        )
        job['result']['cnn'] = {
            'detected': CLASS_NAMES[cnn_info['cnn_class']],
            'confidence': cnn_info['cnn_conf'],
            'top5': cnn_info['cnn_top5'],
        }
        job['result']['cnn_class'] = cnn_info['cnn_class']
        job['result']['cnn_conf'] = cnn_info['cnn_conf']

        job['stage'] = 'lstm'
        lstm_info = await run_lstm()
        job['result']['lstm'] = {
            'forecast': lstm_info['lstm_forecast'].tolist(),
            'context': lstm_info['lstm_ctx'].tolist(),
        }

        job['stage'] = 'fusion'
        fusion_info = await asyncio.get_event_loop().run_in_executor(
            _ML_POOL, run_fusion,
            cnn_info['cnn_embedding'], lstm_info['lstm_ctx'],
            encode_metadata(
                job['meta']['crop_type'], job['meta']['growth_stage'],
                job['meta']['days_since_planting'], job['meta']['days_to_harvest']
            ), job['meta'].get('n_mc_passes', 30)
        )

        job['result']['fusion'] = {
            'top_disease': fusion_info['top_disease'],
            'risk_score': fusion_info['top_risk'],
            'uncertainty': fusion_info['top_uncertainty'],
            'top5': [
                {
                    'disease': DISEASE_CLASSES[i],
                    'probability': float(fusion_info['fusion_calibrated'][i]),
                    'uncertainty': float(fusion_info['fusion_uncertainty'][i]),
                }
                for i in np.argsort(fusion_info['fusion_calibrated'])[::-1][:5]
            ],
        }

        job['stage'] = 'done'
        extras = assemble_forecast_and_yield(cnn_info, fusion_info, job['meta'])
        job['result'].update(extras)

        job['status'] = 'completed'
    except Exception as exc:
        job['status'] = 'failed'
        job['stage']  = 'failed'
        job['error']  = f"{type(exc).__name__}: {exc}"
        print(f"Job {job_id} failed:", exc)


async def job_worker(worker_id: int):
    """Continuously pull jobs from queue for async execution."""
    while True:
        job_id = await job_queue.get()
        try:
            await process_job(job_id)
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def _startup():
    """Warm up models and start worker tasks at app startup."""
    loop = asyncio.get_event_loop()
    def _warm():
        try:
            dummy_feat  = np.zeros((LOOKBACK, N_FEATURES), dtype=np.float32)
            dummy_norm  = weather_scaler.transform(dummy_feat)
            dummy_wx    = torch.tensor(dummy_norm).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lstm_model(dummy_wx)
            print("Warmup pass complete ✓")
        except Exception as e:
            print(f"Warmup warning (non-fatal): {e}")
    await loop.run_in_executor(_ML_POOL, _warm)

    app.state.job_workers = [
        asyncio.create_task(job_worker(i)) for i in range(2)
    ]


@app.on_event("shutdown")
async def _shutdown():
    workers = getattr(app.state, 'job_workers', [])
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

def _run_analyze_sync(
    img_bytes, crop_type, growth_stage,
    days_since_planting, days_to_harvest,
    area_ha, market_price_per_kg,
    n_mc_passes, include_gradcam, lat, lon
):
    """Synchronous full pipeline: CNN → LSTM → Fusion → Yield. 
    Runs in a ThreadPoolExecutor so the event loop stays free."""
    # Decode image
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np    = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_np is None:
        raise ValueError("Could not decode uploaded image")
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Stage 1 — CNN
    resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
    tensor  = val_transform(image=resized)['image'].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = cnn_model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        cnn_class = int(probs.argmax())
        cnn_conf  = float(probs[cnn_class])
        top5 = [
            {'class': CLASS_NAMES[i], 'prob': float(probs[i])}
            for i in probs.argsort(descending=True)[:5]
        ]
        embedding = cnn_model.get_embedding(tensor).squeeze(0).cpu().numpy()

    # Stage 2 — LSTM
    feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
    feat_norm = weather_scaler.transform(feat)
    wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        forecast_raw, wx_ctx, _ = lstm_model(wx_tensor)
    lstm_forecast = forecast_raw[0].cpu().numpy()
    lstm_ctx      = wx_ctx.cpu().numpy()

    # Stage 3 — Fusion
    meta_vec     = encode_metadata(crop_type, growth_stage,
                                   days_since_planting, days_to_harvest)
    fusion_input = torch.tensor(
        np.concatenate([embedding.reshape(1, -1),
                        lstm_ctx.reshape(1, -1),
                        meta_vec.reshape(1, -1)], axis=1),
        dtype=torch.float32
    ).to(DEVICE)
    mean_probs, std_probs = fusion_model.predict_with_uncertainty(
        fusion_input, n_passes=n_mc_passes)
    mean_probs = mean_probs[0].cpu().numpy()
    std_probs  = std_probs[0].cpu().numpy()
    cal_probs  = np.array([
        calibrators[d].predict([mean_probs[i]])[0]
        for i, d in enumerate(DISEASE_CLASSES)
    ])
    top_idx     = int(cal_probs.argmax())
    top_disease = DISEASE_CLASSES[top_idx]
    top_risk    = float(cal_probs[top_idx])
    top_unc     = float(std_probs[top_idx])

    # Stage 4 — Yield & intervention
    meta = {
        'crop_type': crop_type, 'growth_stage': growth_stage,
        'days_since_planting': days_since_planting,
        'days_to_harvest': days_to_harvest,
        'area_ha': area_ha, 'market_price_per_kg': market_price_per_kg,
    }
    cnn_info    = {'cnn_class': cnn_class, 'cnn_conf': cnn_conf}
    fusion_info = {
        'top_idx': top_idx, 'top_disease': top_disease,
        'top_risk': top_risk, 'top_uncertainty': top_unc,
        'fusion_calibrated': cal_probs.tolist(),
        'fusion_uncertainty': std_probs.tolist(),
        'forecast_raw': lstm_forecast.tolist(),
    }
    extras = assemble_forecast_and_yield(cnn_info, fusion_info, meta)

    # Grad-CAM (opt-in)
    gradcam_b64 = None
    if include_gradcam:
        with torch.enable_grad():
            gradcam_b64 = run_gradcam(img_np, cnn_class)

    return {
        'cnn': {
            'detected': CLASS_NAMES[cnn_class],
            'confidence': cnn_conf,
            'top5': top5,
        },
        'fusion': {
            'top_disease': top_disease,
            'risk_score': top_risk,
            'uncertainty': top_unc,
            'top5': [
                {
                    'disease': DISEASE_CLASSES[i],
                    'probability': float(cal_probs[i]),
                    'uncertainty': float(std_probs[i]),
                }
                for i in np.argsort(cal_probs)[::-1][:5]
            ],
        },
        **extras,
        'gradcam': gradcam_b64,
        'location': {'lat': lat, 'lon': lon},
        'timestamp': datetime.now().isoformat(),
    }

@app.post("/analyze")
async def analyze(
    file                : UploadFile = File(...),
    lat                 : float = 18.5204,
    lon                 : float = 73.8567,
    crop_type           : str   = "Tomato",
    growth_stage        : str   = "fruiting",
    days_since_planting : int   = 75,
    days_to_harvest     : int   = 20,
    area_ha             : float = 1.0,
    market_price_per_kg : float = 25.0,
    # Reduced from 30 → 10.  Still gives reliable uncertainty bands,
    # cuts MC-dropout time by ~3×.  Flutter can pass 30 explicitly if needed.
    n_mc_passes         : int   = 10,
    # Set to False by default — Flutter requests Grad-CAM only when the
    # user taps "Show Heatmap", not on every analysis.
    include_gradcam     : bool  = False,
):
    img_bytes = await file.read()
 
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _run_analyze_sync,
        img_bytes,
        crop_type,
        growth_stage,
        days_since_planting,
        days_to_harvest,
        area_ha,
        market_price_per_kg,
        n_mc_passes,
        include_gradcam,
        lat,
        lon,
    )
    return result


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')
    response = {
        'job_id': job_id,
        'status': job['status'],
        'stage': job.get('stage', 'unknown'),
        'result': job.get('result', {}),
    }
    if job.get('error'):
        response['error'] = job['error']
    return response


@app.get("/explain/{job_id}")
async def explain(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job ID not found')
    if job['status'] != 'completed':
        raise HTTPException(status_code=400,
                            detail='Job not completed yet: explain available only after completion')

    cnn_class_name = job['result']['cnn']['detected']
    cnn_class_idx  = CLASS_NAMES.index(cnn_class_name)

    img_np = job.get('input_image_np')
    if img_np is None:
        raise HTTPException(status_code=410,
                            detail='Input image not available for explanation')

    gradcam_b64 = await asyncio.get_event_loop().run_in_executor(
        _ML_POOL, run_gradcam, img_np, cnn_class_idx
    )

    return {
        'job_id': job_id,
        'gradcam': gradcam_b64,
        'target_class': cnn_class_name,
    }


@app.get("/config")
def get_config():
    return {
        "crop_types"    : CROP_TYPES,
        "growth_stages" : GROWTH_STAGES,
        "disease_classes": DISEASE_CLASSES,
        "class_names"   : CLASS_NAMES,
    }
 
 
@app.get("/forecast")
async def get_forecast(lat: float = 18.5204, lon: float = 73.8567,
                       top_n: int = 10):
    """7-day disease risk forecast for a GPS location."""
    loop = asyncio.get_event_loop()

    def _run():
        feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
        feat_norm = weather_scaler.transform(feat)
        wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
        lstm_model.eval()
        with torch.no_grad():
            forecast_raw, _, _ = lstm_model(wx_tensor)
        forecast_np  = forecast_raw[0].cpu().numpy()
        peak_risk_np = forecast_np.max(axis=0)
        ranked_idx   = peak_risk_np.argsort()[::-1][:top_n]
        forecast_dates = [
            (datetime.now() + timedelta(days=i+1)).strftime('%a %d %b')
            for i in range(FORECAST)
        ]
        return {
            'location' : {'lat': lat, 'lon': lon},
            'dates'    : forecast_dates,
            'diseases' : [
                {
                    'disease'  : DISEASE_CLASSES[i],
                    'peak_risk': float(peak_risk_np[i]),
                    'daily'    : [float(forecast_np[d, i]) for d in range(FORECAST)],
                    'level'    : 'HIGH' if peak_risk_np[i] > 0.6
                                 else 'MODERATE' if peak_risk_np[i] > 0.3
                                 else 'LOW',
                }
                for i in ranked_idx
            ],
            'timestamp': datetime.now().isoformat(),
        }

    try:
        return await loop.run_in_executor(_ML_POOL, _run)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Forecast failed: {type(e).__name__}: {e}")
 
 
@app.get("/historical")
async def get_historical(disease: str, days_back: int = 365):
    """Historical DCWS risk scores for a disease."""
    if disease not in dcws_df.columns:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Disease '{disease}' not found. "
                f"Available (first 5): {list(dcws_df.columns[:5])}"
            )
        )
    loop   = asyncio.get_event_loop()
    series = await loop.run_in_executor(
        None, lambda: dcws_df[disease].tail(days_back))
    return {

        'disease': disease,
        'dates'  : series.index.strftime('%Y-%m-%d').tolist(),
        'scores' : series.round(4).tolist(),
        'mean'   : float(series.mean()),
        'max'    : float(series.max()),
    }

@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.get("/compare")
async def compare_crops(
    lat          : float = 18.5204,
    lon          : float = 73.8567,
    growth_stage : str   = "fruiting",
    days_to_harvest: int = 30,
):
    """Compares risk across all crop types for current weather."""
    loop = asyncio.get_event_loop()

    def _run():
        feat      = weather_df[WEATHER_FEATURES].values[-LOOKBACK:].astype(np.float32)
        feat_norm = weather_scaler.transform(feat)
        wx_tensor = torch.tensor(feat_norm).unsqueeze(0).to(DEVICE)
        lstm_model.eval()
        with torch.no_grad():
            _, wx_ctx, _ = lstm_model(wx_tensor)
        wx_ctx_np = wx_ctx.cpu().numpy()

        results = {}
        for crop in CROP_TYPES:
            meta = encode_metadata(crop, growth_stage, 60, days_to_harvest
                                   ).reshape(1, -1)
            cnn_zero     = np.zeros((1, CNN_DIM), dtype=np.float32)
            fusion_input = torch.tensor(
                np.concatenate([cnn_zero, wx_ctx_np, meta], axis=1),
                dtype=torch.float32
            ).to(DEVICE)
            fusion_model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(
                    fusion_model(fusion_input))[0].cpu().numpy()
            cal_probs = np.array([
                calibrators[d].predict([probs[i]])[0]
                for i, d in enumerate(DISEASE_CLASSES)
            ])
            results[crop] = {
                'max_risk'   : float(cal_probs.max()),
                'top_disease': DISEASE_CLASSES[int(cal_probs.argmax())],
                'risk_level' : 'HIGH' if cal_probs.max() > 0.6
                               else 'MODERATE' if cal_probs.max() > 0.3
                               else 'LOW',
            }
        return {'crops': results, 'timestamp': datetime.now().isoformat()}

    try:
        return await loop.run_in_executor(_ML_POOL, _run)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Comparison failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",       # bind to all interfaces so ngrok can reach it
        port=8000,
        reload=False,          # keep False in production
        workers=1,             # single worker keeps ML models in one process
        timeout_keep_alive=65, # longer than Flutter's 30 s connect timeout
        log_level="info",
    )