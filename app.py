import os
import base64
import webbrowser
import threading
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
from google import genai  # <-- ADD THIS IMPORT
import uvicorn
import io

load_dotenv()

app = FastAPI(title="DermaSafe AI API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Resolution
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY environment variable is missing.")
    return Groq(api_key=api_key)


# Add helper for Gemini client setup
def get_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY environment variable is missing.")
    
    # Initialize directly with the key
    return genai.Client(api_key=api_key)
# =========================================================================
# EMBEDDED BIOCHEMICAL CLINICAL INTERACTION DATASET
# =========================================================================
INGREDIENT_MUTUAL_CONFLICTS = [
    {"a": ["retinol", "retinal", "retinyl palmitate"], "b": ["isotretinoin", "tretinoin", "adapalene", "tazarotene"], "msg": "Additive Retinoid Overload: Extreme risk of skin peeling, dryness, and barrier breakdown.", "sev": "CRITICAL"},
    {"a": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate"], "b": ["benzoyl peroxide"], "msg": "Irritation risk increases. Note: Benzoyl peroxide can degrade/inactivate specific unstable retinoids like tretinoin if mixed.", "sev": "HIGH WARNING"},
    {"a": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate", "glycolic acid", "lactic acid", "mandelic acid", "salicylic acid", "betaine salicylate"], "b": ["physical exfoliant", "scrub", "walnut shell", "beads"], "msg": "Exfoliant Stacking: Mixing clinical chemical cell-turnover agents with friction scrubs destroys structural lipid layers.", "sev": "CRITICAL"},
    {"a": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate"], "b": ["glycolic acid", "lactic acid", "mandelic acid"], "msg": "Retinoid + AHA Collision: Accelerates peeling, intense surface redness, and cutaneous burning.", "sev": "HIGH WARNING"},
    {"a": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate"], "b": ["salicylic acid", "betaine salicylate"], "msg": "Retinoid + BHA Collision: Deep lipid breakdown causing flaking, profound tightness, and inflammation.", "sev": "HIGH WARNING"},
    {"a": ["benzoyl peroxide"], "b": ["hydroquinone"], "msg": "Reactivity Conflict: Can cause temporary dark orange/brown skin staining when combined.", "sev": "WARNING"},
    {"a": ["benzoyl peroxide"], "b": ["sulfur"], "msg": "Severe moisture vacuum: Triggers extreme epidermal dryness and localized irritation.", "sev": "WARNING"},
    {"a": ["benzoyl peroxide"], "b": ["salicylic acid", "betaine salicylate"], "msg": "Severe multi-pathway acne treatment stack: Elevates cumulative irritation thresholds.", "sev": "WARNING"},
    {"a": ["sulfur"], "b": ["salicylic acid", "betaine salicylate"], "msg": "Synergistic stripping: Induces rapid lipid extraction and flaking.", "sev": "WARNING"},
    {"a": ["hydroquinone"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Aggressive pathway stack: Amplifies risk of contact dermatitis.", "sev": "HIGH WARNING"},
    {"a": ["kojic acid"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Melanogenesis inhibitor + Retinoid: High chance of transient irritation or stinging.", "sev": "WARNING"},
    {"a": ["arbutin"], "b": ["glycolic acid", "lactic acid", "salicylic acid"], "msg": "Strong exfoliant stacking with arbutin may compromise skin tolerances, raising irritation risk.", "sev": "WARNING"},
    {"a": ["azelaic acid"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Co-application causes prominent localized dryness and transient redness.", "sev": "WARNING"},
    {"a": ["azelaic acid"], "b": ["glycolic acid", "lactic acid", "mandelic acid"], "msg": "Dual acid load: Multiplies surface irritation potential.", "sev": "WARNING"},
    {"a": ["tranexamic acid"], "b": ["glycolic acid", "lactic acid", "salicylic acid"], "msg": "Mild potential for heightened barrier vulnerability when paired with chemical peeling agents.", "sev": "WARNING"},
    {"a": ["ascorbic acid", "vitamin c"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "pH/Irritation Sensitivity: Applying pure Vitamin C alongside retinoids can cause irritation for sensitive skin.", "sev": "WARNING"},
    {"a": ["ascorbic acid", "vitamin c"], "b": ["glycolic acid", "lactic acid"], "msg": "Low pH compounding: Increases baseline stinging and temporary redness.", "sev": "WARNING"},
    {"a": ["sodium ascorbyl phosphate"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Mild risk of multi-active irritation.", "sev": "WARNING"},
    {"a": ["copper peptide"], "b": ["ascorbic acid", "vitamin c"], "msg": "Antioxidant complex conflict: May reduce peptide stability and overall effectiveness.", "sev": "WARNING"},
    {"a": ["copper peptide"], "b": ["glycolic acid", "lactic acid", "salicylic acid"], "msg": "Acidic cleavage: Low pH environments break down copper peptide chains.", "sev": "WARNING"},
    {"a": ["niacinamide"], "b": ["glycolic acid", "lactic acid", "ascorbic acid"], "msg": "Acid flushing: Rare temporary vasodilation/flushing reaction due to conversion to nicotinic acid at low pH.", "sev": "NEUTRAL / INFO"},
    {"a": ["phenoxyethanol", "fragrance", "parfum", "essential oil"], "b": ["damaged barrier", "compromised barrier"], "msg": "Preservative/Aromatic sensitivity: Heightened irritation potential on weak skin barriers.", "sev": "WARNING"},
    {"a": ["citrus oil", "lemon oil", "bergamot oil", "lime oil"], "b": ["uv", "sun"], "msg": "Phototoxicity Risk: Citrus oils under UV light can trigger severe blistering hyperpigmentation burns.", "sev": "HIGH WARNING"},
    {"a": ["tea tree oil"], "b": ["benzoyl peroxide"], "msg": "Dual antimicrobial desiccation: Causes elevated trans-epidermal water loss.", "sev": "WARNING"},
    {"a": ["papain", "bromelain"], "b": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Enzymatic exfoliation stacked onto accelerated cell turnover speeds up barrier stripping.", "sev": "HIGH WARNING"}
]

MEDICATION_CONFLICTS = {
    "Isotretinoin (Accutane)": [
        {"triggers": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate"], "msg": "CRITICAL CONFLICT: Systemic isotretinoin combined with topicals creates severe skin thinning and peeling.", "sev": "CRITICAL"},
        {"triggers": ["glycolic acid", "lactic acid", "mandelic acid"], "msg": "HIGH RISK: Accutane skin is fragile; AHAs can cause chemical burns.", "sev": "CRITICAL"},
        {"triggers": ["salicylic acid", "betaine salicylate"], "msg": "HIGH RISK: Excessive lipid barrier stripping on dry Accutane skin.", "sev": "CRITICAL"},
        {"triggers": ["benzoyl peroxide"], "msg": "Severe irritation and painful dry patches.", "sev": "HIGH WARNING"},
        {"triggers": ["physical exfoliant", "scrub"], "msg": "Mechanical skin tearing risk due to extreme thinning.", "sev": "CRITICAL"}
    ],
    "Tretinoin (Topical)": [
        {"triggers": ["benzoyl peroxide"], "msg": "Co-application risk: Increased irritation; BP can degrade unencapsulated tretinoin if applied at the exact same time.", "sev": "HIGH WARNING"},
        {"triggers": ["glycolic acid", "lactic acid"], "msg": "Excessive peeling, redness, and micro-tears.", "sev": "HIGH WARNING"},
        {"triggers": ["salicylic acid"], "msg": "Severe localized dehydration and stinging.", "sev": "HIGH WARNING"}
    ],
    "Adapalene (Differin)": [
        {"triggers": ["glycolic acid", "lactic acid", "salicylic acid"], "msg": "Compounded epidermal irritation and dry flaking.", "sev": "WARNING"}
    ],
    "Tazarotene": [
        {"triggers": ["glycolic acid", "lactic acid", "salicylic acid"], "msg": "Tazarotene is highly potent; alpha/beta hydroxy acids will trigger severe chemical peeling and redness.", "sev": "CRITICAL"}
    ],
    "Doxycycline (Oral)": [
        {"triggers": ["glycolic acid", "lactic acid", "mandelic acid"], "msg": "Photosensitivity Stacking: Doxycycline causes systemic sun sensitivity; AHAs amplify it.", "sev": "HIGH WARNING"},
        {"triggers": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Heightened phototoxicity and generalized skin irritation.", "sev": "HIGH WARNING"}
    ],
    "Minocycline (Oral)": [
        {"triggers": ["glycolic acid", "lactic acid"], "msg": "Increased systemic and localized sunburn risks.", "sev": "HIGH WARNING"}
    ],
    "Hydrocortisone (Topical Steroid)": [
        {"triggers": ["glycolic acid", "lactic acid"], "msg": "Steroids thin the skin; AHAs on thinned skin can cause chemical burns.", "sev": "HIGH WARNING"}
    ],
    "Clobetasol (Topical Steroid)": [
        {"triggers": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Compounded cutaneous thinning, telangiectasia risk, and severe irritation.", "sev": "CRITICAL"}
    ],
    "Tacrolimus (Protopic)": [
        {"triggers": ["glycolic acid", "lactic acid"], "msg": "Intense burning and flare-up sensations.", "sev": "HIGH WARNING"}
    ],
    "Pimecrolimus (Elidel)": [
        {"triggers": ["retinol", "tretinoin", "adapalene", "tazarotene"], "msg": "Increased localized burning and dynamic barrier sensitivity.", "sev": "HIGH WARNING"}
    ]
}

CONDITION_FILTER_RULES = {
    "Pregnancy-Safe Filter": {
        "triggers": ["retinol", "tretinoin", "adapalene", "tazarotene", "retinal", "retinyl palmitate", "hydroquinone", "oxybenzone", "salicylic acid"],
        "msg": "Teratogenic / High Absorption Risk: Retinoids, Hydroquinone, and Oxybenzone are explicitly flagged during pregnancy. Limit leave-on Salicylic Acid to low percentages.",
        "sev": "CRITICAL"
    },
    "Breastfeeding Filter": {
        "triggers": ["tretinoin", "tazarotene", "hydroquinone"],
        "msg": "Systemic transmission caution: Avoid powerful prescription retinoids or skin lighteners unless explicitly cleared.",
        "sev": "HIGH WARNING"
    },
    "Rosacea Filter": {
        "triggers": ["glycolic acid", "lactic acid", "benzoyl peroxide", "retinol", "tretinoin", "adapalene", "tazarotene", "fragrance", "alcohol denat"],
        "msg": "Vascular Hyper-reactivity: Triggers instant flushing, stinging, and permanent capillary dilation.",
        "sev": "CRITICAL"
    },
    "Eczema Filter": {
        "triggers": ["fragrance", "parfum", "alcohol denat", "sd alcohol", "isopropyl alcohol", "glycolic acid", "lactic acid", "retinol", "tretinoin"],
        "msg": "Barrier Disruption: Drives immediate contact dermatitis, allergic flares, or stinging patches.",
        "sev": "CRITICAL"
    },
    "Psoriasis Filter": {
        "triggers": ["physical exfoliant", "scrub", "glycolic acid", "salicylic acid"],
        "msg": "Koebner Phenomenon Risk: Aggressive scrubbing or strong peeling actions can trigger psoriasis flares on clear skin areas.",
        "sev": "HIGH WARNING"
    },
    "Aspirin Allergy Filter": {
        "triggers": ["salicylic acid", "betaine salicylate", "salix alba", "willow bark"],
        "msg": "Cross-Reactivity Alert: Salicylates share structural paths with aspirin and can cause hives or breathing trouble.",
        "sev": "CRITICAL"
    },
    "Photosensitive Skin Filter": {
        "triggers": ["glycolic acid", "lactic acid", "retinol", "tretinoin", "adapalene", "tazarotene"],
        "msg": "UV Vulnerability: Eliminates stratifying barrier protection, dramatically increasing the risk of deep sunburns.",
        "sev": "HIGH WARNING"
    }
}

INTERNATIONAL_BANNED_INGREDIENTS = [
    {
        "triggers": ["lilial", "butylphenyl methylpropional"],
        "region": "EU & UK (Annex II)",
        "msg": "Banned in EU/UK: Reprotoxic chemical (may harm fertility and fetal development). Banned since March 2022.",
        "sev": "CRITICAL"
    },
    {
        "triggers": ["lyral", "hydroxyisohexyl 3-cyclohexene carboxaldehyde"],
        "region": "EU & UK",
        "msg": "Banned in EU/UK: Severe contact allergen and skin sensitizer. Banned since 2021.",
        "sev": "CRITICAL"
    },
    {
        "triggers": ["isobutylparaben", "isopropylparaben", "phenylparaben", "benzylparaben", "pentylparaben"],
        "region": "European Union (EU)",
        "msg": "Banned in EU: Endocrine disruptors causing hormonal imbalances and reproductive harm.",
        "sev": "CRITICAL"
    },
    {
        "triggers": ["methylisothiazolinone", "mit"],
        "region": "EU & UK (Leave-on products)",
        "msg": "Banned in EU/UK in leave-on products due to widespread risk of contact allergy and eczema flares.",
        "sev": "HIGH WARNING"
    },
    {
        "triggers": ["hydroquinone"],
        "region": "EU, UK, Japan, FDA OTC Ban",
        "msg": "Banned OTC in EU/UK/US: Exogenous ochronosis risk and cytotoxicity concerns.",
        "sev": "HIGH WARNING"
    },
    {
        "triggers": ["triclosan", "triclocarban"],
        "region": "EU & FDA OTC",
        "msg": "Banned/Restricted: Endocrine disruption and bacterial resistance acceleration.",
        "sev": "HIGH WARNING"
    },
    {
        "triggers": ["cyclotetrasiloxane", "d4"],
        "region": "European Union (EU)",
        "msg": "Banned/Restricted in EU: Toxic to human reproduction and bioaccumulative.",
        "sev": "HIGH WARNING"
    },
    {
        "triggers": ["quaternium-15", "dmdm hydantoin", "methylene glycol"],
        "region": "European Union (EU)",
        "msg": "Restricted/Banned in EU: Formaldehyde-releasing preservative linked to severe allergies.",
        "sev": "HIGH WARNING"
    },
    {
        "triggers": ["lead acetate"],
        "region": "EU, Canada, FDA (Phased)",
        "msg": "Banned globally: Heavy metal neurotoxin previously found in hair dyes.",
        "sev": "CRITICAL"
    }
]

GENERAL_HAZARDS = {
    "formaldehyde": "Carcinogen and severe contact allergen.",
    "paraben": "Potential endocrine disruptor concerns.",
    "triclosan": "Environmental toxin and endocrine disruptor.",
    "phthalate": "Potential hormone mimic and reproductive system irritant.",
    "sodium lauryl sulfate": "Aggressive surfactant that can strip the natural skin barrier.",
    "toluene": "Volatile nail/cosmetic solvent linked to neurotoxicity.",
    "coal tar": "Known carcinogen used in scaling conditions; requires medical guidance."
}

# --- Pydantic Data Models (With Robust Default Values) ---
class SafetyAnalysisRequest(BaseModel):
    ingredients: str
    medications: List[str] = []
    conditions: List[str] = []
    skin_type: Optional[str] = ""

class RoutineRecommendationRequest(BaseModel):
    routine_items: List[dict] = []
    medications: List[str] = []
    conditions: List[str] = []
    skin_type: Optional[str] = ""

class GenerateCustomRoutineRequest(BaseModel):
    skin_type: Optional[str] = ""
    goals: Optional[str] = ""
    medications: List[str] = []
    conditions: List[str] = []

class ProductSearchRequest(BaseModel):
    product_name: str

class ProductOverviewRequest(BaseModel):
    product_name: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    skin_type: Optional[str] = ""
    medications: List[str] = []
    conditions: List[str] = []


# Serve Landing Page as Home
@app.get("/", response_class=HTMLResponse)
def get_landing():
    landing_path = os.path.join(BASE_DIR, "landing.html")
    if os.path.exists(landing_path):
        with open(landing_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DermaSafe AI Landing Page (landing.html not found)</h1>"

@app.get("/app", response_class=HTMLResponse)
def get_app():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DermaSafe AI Dashboard (index.html not found)</h1>"

# 1. Product Search API
@app.post("/api/search-product")
def search_product(req: ProductSearchRequest):
    client = get_groq_client()
    try:
        prompt = (
            f"Provide the exact full INCI ingredient list for the cosmetic product: '{req.product_name}'. "
            f"Return ONLY the comma-separated ingredient list without conversational text or bullet points."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        return {"ingredients": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2.5 Product Overview APIs (Text & Vision)
@app.post("/api/product-overview")
def product_overview(req: ProductOverviewRequest):
    client = get_groq_client()
    try:
        prompt = (
            f"Provide a clear, helpful overview of the cosmetic product: '{req.product_name}'.\n"
            f"Include:\n"
            f"1. What the product is.\n"
            f"2. Primary function and main skin benefits.\n"
            f"3. Ideal target skin types/concerns.\n"
            f"4. How and when it is typically used in a routine.\n\n"
            f"Format with clean Markdown bullet points."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        return {"overview": response.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. Vision OCR API
# 1. Vision OCR Extraction Endpoint (Powered by Gemini)
@app.post("/api/vision-extract")
async def vision_extract(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        client = get_gemini_client()
        prompt = "Extract all ingredient text from this image. Output ONLY a comma-separated list of ingredients cleanly so they can be processed."
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[image, prompt]
        )
        return {"ingredients": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. Product Overview Vision Endpoint (Powered by Gemini)
@app.post("/api/product-overview-vision")
async def product_overview_vision(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        client = get_gemini_client()
        prompt = (
            "Identify this skincare/cosmetic product from the image and provide a concise overview:\n"
            "1. Product Name & Brand\n"
            "2. What the product is and its main function\n"
            "3. Key benefits and targeted skin concerns\n"
            "4. Application guidance.\n\n"
            "Format in clean Markdown bullet points."
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[image, prompt]
        )
        return {"overview": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Safety Analysis Engine Endpoint
@app.post("/api/analyze-safety")
def analyze_safety(req: SafetyAnalysisRequest):
    conflicts = []
    ingredients_lower = req.ingredients.lower()

    # 1. Ingredient Synergy Conflict
    for rule in INGREDIENT_MUTUAL_CONFLICTS:
        matched_a = [item.upper() for item in rule["a"] if item in ingredients_lower]
        matched_b = [item.upper() for item in rule["b"] if item in ingredients_lower]
        if matched_a and matched_b:
            conflicts.append({
                "Type": "Ingredient Synergy Conflict",
                "Flagged Ingredients": f"{', '.join(matched_a)} + {', '.join(matched_b)}",
                "Source / Condition": "Interaction Matrix Match",
                "Conflict Details": rule["msg"],
                "Severity": rule["sev"]
            })

    # 2. Prescription Conflict
    for med in req.medications:
        if med in MEDICATION_CONFLICTS:
            for rule in MEDICATION_CONFLICTS[med]:
                matched_triggers = [trigger.upper() for trigger in rule["triggers"] if trigger in ingredients_lower]
                if matched_triggers:
                    conflicts.append({
                        "Type": "Prescription Conflict",
                        "Flagged Ingredients": ", ".join(matched_triggers),
                        "Source / Condition": f"Medication: {med}",
                        "Conflict Details": rule["msg"],
                        "Severity": rule["sev"]
                    })

    # 3. Health Strategy Filter
    for cond in req.conditions:
        if cond in CONDITION_FILTER_RULES:
            rule = CONDITION_FILTER_RULES[cond]
            matched_triggers = [trigger.upper() for trigger in rule["triggers"] if trigger in ingredients_lower]
            if matched_triggers:
                conflicts.append({
                    "Type": "Health Strategy Filter",
                    "Flagged Ingredients": ", ".join(matched_triggers),
                    "Source / Condition": f"Condition: {cond}",
                    "Conflict Details": rule["msg"],
                    "Severity": rule["sev"]
                })

    # 4. International Regulatory Banned Check
    for ban in INTERNATIONAL_BANNED_INGREDIENTS:
        matched = [trigger.upper() for trigger in ban["triggers"] if trigger in ingredients_lower]
        if matched:
            conflicts.append({
                "Type": "Global Regulatory Banned",
                "Flagged Ingredients": ", ".join(matched),
                "Source / Condition": f"Regulatory Standard: {ban['region']}",
                "Conflict Details": ban["msg"],
                "Severity": ban["sev"]
            })

    # Calculate Safety Score Percentage
    critical_count = len([c for c in conflicts if c["Severity"] == "CRITICAL"])
    warning_count = len([c for c in conflicts if c["Severity"] in ["HIGH WARNING", "WARNING"]])
    
    s1 = max(0, 100 - (critical_count * 20))
    s2 = max(0, 100 - (len([c for c in conflicts if c["Type"] == "Prescription Conflict"]) * 30))
    s3 = max(0, 100 - (len([c for c in conflicts if c["Type"] == "Health Strategy Filter"]) * 25))
    
    safety_score = round((s1 + s2 + s3) / 3)

    # Generate AI Product Recommendations if Safety Score < 70%
    recommendations = ""
    if safety_score < 70:
        try:
            client = get_groq_client()
            flagged = list(set([c["Flagged Ingredients"] for c in conflicts]))
            prompt = (
                f"The user analyzed a product with ingredients that received a low safety score of {safety_score}%.\n"
                f"Flagged conflicts / harmful ingredients: {', '.join(flagged)}.\n"
                f"User Profile - Skin: {req.skin_type or 'Not specified'}, Medications: {', '.join(req.medications) or 'None'}, Conditions: {', '.join(req.conditions) or 'None'}.\n\n"
                f"Please recommend 2-3 real, safe alternative skincare products that feature similar key beneficial ingredients and cosmetic effects, but STRICTLY exclude the flagged harmful or conflicting ingredients. Format with concise bullet points."
            )
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            recommendations = resp.choices[0].message.content.strip()
        except Exception:
            recommendations = "Alternative recommendations could not be fetched at this time."

    return {
        "conflicts": conflicts,
        "safety_score": safety_score,
        "recommendations": recommendations
    }

# 3.5 Routine AI Recommendations Endpoint
@app.post("/api/routine-recommendations")
def get_routine_recommendations(req: RoutineRecommendationRequest):
    client = get_groq_client()
    try:
        items_desc = "\n".join([f"- Product: {item.get('name', 'Unknown')} (Actives: {item.get('active', 'None specified')})" for item in req.routine_items])
        meds_desc = ", ".join(req.medications) if req.medications else "None"
        conds_desc = ", ".join(req.conditions) if req.conditions else "None"
        skin_desc = req.skin_type if req.skin_type else "Not specified"

        prompt = (
            f"You are a helpful, easy-to-understand skincare assistant.\n"
            f"Review this list of products:\n{items_desc}\n\n"
            f"User Info:\n"
            f"- Skin Type: {skin_desc}\n"
            f"- Active Medications: {meds_desc}\n"
            f"- Health Conditions/Filters: {conds_desc}\n\n"
            f"Instructions:\n"
            f"Provide a brief, plain-English routine guide. Avoid heavy scientific jargon.\n"
            f"Do NOT use markdown headers or '###' symbols in your response.\n\n"
            f"Format strictly using exact titles:\n"
            f"☀️ AM Routine (Morning):\n"
            f"- List products in application order with 1 short line explaining why.\n\n"
            f"🌙 PM Routine (Evening):\n"
            f"- List products in application order with 1 short line explaining why.\n\n"
            f"⚠️ Simple Safety Rules & Conflicts:\n"
            f"- Mention any ingredients that shouldn't be used together or on the same day in 1-2 simple sentences.\n\n"
            f"Keep explanations super concise, clear, and friendly. Do not include intro or outro conversational fluff."
        )

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=450
        )
        return {"recommendation": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3.6 Generate Custom Routine Endpoint
@app.post("/api/generate-custom-routine")
def generate_custom_routine(req: GenerateCustomRoutineRequest):
    client = get_groq_client()
    try:
        meds_desc = ", ".join(req.medications) if req.medications else "None"
        conds_desc = ", ".join(req.conditions) if req.conditions else "None"
        skin_desc = req.skin_type if req.skin_type else "Not specified"
        goals_desc = req.goals if req.goals else "General skin health and hydration"

        prompt = (
            f"You are an expert cosmetic biochemist and board-certified dermatologist.\n"
            f"Build a complete, customized skincare routine from scratch for this user profile:\n"
            f"- Skin Type: {skin_desc}\n"
            f"- Skin Goals / Concerns: {goals_desc}\n"
            f"- Active Medications: {meds_desc}\n"
            f"- Health/Skin Conditions & Filters: {conds_desc}\n\n"
            f"Instructions:\n"
            f"1. Recommend specific real-world products for both Morning (AM) and Evening (PM) routines.\n"
            f"2. List step-by-step application order (e.g., Step 1: Cleanser, Step 2: Serum...).\n"
            f"3. Explain why each product/ingredient was selected based on their profile and goals.\n"
            f"4. Strictly avoid ingredients that conflict with their listed medications or health filters.\n\n"
            f"Format with clean Markdown headings (### AM Routine, ### PM Routine, ### Key Ingredients & Benefits)."
        )

        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=850
        )
        return {"routine": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Chatbot Endpoint
@app.post("/api/chat")
def chat(req: ChatRequest):
    client = get_groq_client()
    try:
        med_context = ", ".join(req.medications) if req.medications else "None"
        cond_context = ", ".join(req.conditions) if req.conditions else "None"
        skin_context = req.skin_type if req.skin_type else "Not specified"
        
        system_prompt = f"Expert biochemist assistant. User skin: {skin_context}. Medications: {med_context}. Active Conditions/Filters applied: {cond_context}."
        
        api_messages = [{"role": "system", "content": system_prompt}] + [{"role": m.role, "content": m.content} for m in req.messages]
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
            temperature=0.3,
            max_tokens=1024
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper to launch browser safely after server starts
def open_browser():
    if os.getenv("RENDER") or os.getenv("ENVIRONMENT") == "production":
        return
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    if os.getenv("RENDER") or os.getenv("ENVIRONMENT") == "production":
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        threading.Thread(target=open_browser, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=8000)