import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import json

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

DATA = None

# Branch mappings from frontend ID to backend dataset codes
BRANCH_MAPPING = {
    'CSE': ['CSE'],
    'CSE-AI': ['CSM', 'AIM', 'AI'],
    'CSE-DS': ['CSD', 'AID'],
    'CSE-CS': ['CSC', 'CSO', 'CIC'],
    'IT': ['INF', 'CSI'],
    'ECE': ['ECE'],
    'EEE': ['EEE'],
    'MECH': ['MEC'],
    'CIVIL': ['CIV'],
    'CHEM': ['CHE']
}

def load_and_process_data():
    files = {
        1: 'phase 1 2026.xlsx',
        2: 'phase 2 2026.xlsx',
        3: 'phase 3 2026.xlsx'
    }
    
    dfs = []
    for phase, file in files.items():
        try:
            df = pd.read_excel(file)
            
            # Clean column names
            df.columns = [str(c).replace('\n', '_') if 'BOYS' in str(c) or 'GIRLS' in str(c) else str(c).replace('\n', ' ') for c in df.columns]
            
            # Mapping to target names
            col_map = {
                'Inst Code': 'institute_code',
                'Institute Name': 'institute_name',
                'Place': 'place',
                'College Type': 'college_type',
                'Branch Code': 'branch_code',
                'Branch Name': 'branch_name'
            }
            df = df.rename(columns=col_map)
            
            # Extract categories
            categories = [
                'OC_BOYS', 'OC_GIRLS', 'BC_A_BOYS', 'BC_A_GIRLS', 'BC_B_BOYS', 'BC_B_GIRLS', 
                'BC_C_BOYS', 'BC_C_GIRLS', 'BC_D_BOYS', 'BC_D_GIRLS', 'BC_E_BOYS', 'BC_E_GIRLS', 
                'SC_I_BOYS', 'SC_I_GIRLS', 'SC_II_BOYS', 'SC_II_GIRLS', 'SC_III_BOYS', 'SC_III_GIRLS', 
                'ST_BOYS', 'ST_GIRLS', 'EWS_BOYS', 'EWS_GIRLS'
            ]
            
            # Filter existing category columns
            existing_cats = [c for c in categories if c in df.columns]
            
            id_vars = ['institute_code', 'institute_name', 'place', 'college_type', 'branch_code', 'branch_name']
            
            # keep only required columns
            keep_cols = id_vars + existing_cats
            df = df[[c for c in keep_cols if c in df.columns]].copy()
            
            df['phase'] = phase
            
            # Melt
            df_melted = pd.melt(df, id_vars=id_vars + ['phase'], value_vars=existing_cats, var_name='category', value_name='cutoff_rank')
            
            # Clean cutoff_rank
            df_melted['cutoff_rank'] = pd.to_numeric(df_melted['cutoff_rank'], errors='coerce')
            df_melted = df_melted.dropna(subset=['cutoff_rank'])
            
            dfs.append(df_melted)
        except Exception as e:
            print(f"Error loading {file}: {e}")
        
    if not dfs:
        print("No data loaded!")
        return pd.DataFrame()
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Pivot to get phases as columns
    pivot_df = merged_df.pivot_table(
        index=['institute_code', 'institute_name', 'place', 'college_type', 'branch_code', 'branch_name', 'category'],
        columns='phase',
        values='cutoff_rank'
    ).reset_index()
    
    # Ensure phase columns exist
    for p in [1, 2, 3]:
        if p not in pivot_df.columns:
            pivot_df[p] = np.nan
            
    # Rename phase columns
    pivot_df = pivot_df.rename(columns={1: 'phase1_cutoff', 2: 'phase2_cutoff', 3: 'phase3_cutoff'})
    
    pivot_df['avg_cutoff'] = pivot_df[['phase1_cutoff', 'phase2_cutoff', 'phase3_cutoff']].mean(axis=1)
    
    pivot_df['cutoff_change_1_to_2'] = pivot_df['phase2_cutoff'] - pivot_df['phase1_cutoff']
    pivot_df['cutoff_change_2_to_3'] = pivot_df['phase3_cutoff'] - pivot_df['phase2_cutoff']
    
    def get_trend(c1, c2, c3):
        vals = [v for v in [c1, c2, c3] if not pd.isna(v)]
        if len(vals) < 2:
            return "STABLE"
        last = vals[-1]
        prev = vals[-2]
        if last < prev: return "TIGHTENING"
        if last > prev: return "RELAXING"
        return "STABLE"
        
    pivot_df['trend_direction'] = pivot_df.apply(lambda row: get_trend(row['phase1_cutoff'], row['phase2_cutoff'], row['phase3_cutoff']), axis=1)
    
    pivot_df['volatility_score'] = pivot_df[['phase1_cutoff', 'phase2_cutoff', 'phase3_cutoff']].std(axis=1).fillna(0)
    
    def predict_cutoff(p1, p2, p3):
        weights = {1: 0.2, 2: 0.3, 3: 0.5}
        total_w = 0
        val = 0
        if not pd.isna(p1): 
            val += p1 * weights[1]
            total_w += weights[1]
        if not pd.isna(p2):
            val += p2 * weights[2]
            total_w += weights[2]
        if not pd.isna(p3):
            val += p3 * weights[3]
            total_w += weights[3]
            
        if total_w == 0: return np.nan
        return val / total_w
        
    pivot_df['predicted_cutoff'] = pivot_df.apply(lambda row: predict_cutoff(row['phase1_cutoff'], row['phase2_cutoff'], row['phase3_cutoff']), axis=1)
    
    def calculate_confidence(p1, p2, p3, vol):
        conf = 100
        if pd.isna(p1): conf -= 20
        if pd.isna(p2): conf -= 20
        if pd.isna(p3): conf -= 30
        
        vol_penalty = min(30, (vol / 5000) * 30)
        conf -= vol_penalty
        return max(0, min(100, conf))
        
    pivot_df['confidence_score'] = pivot_df.apply(lambda row: calculate_confidence(row['phase1_cutoff'], row['phase2_cutoff'], row['phase3_cutoff'], row['volatility_score']), axis=1)
    
    return pivot_df

print("Loading and processing data...")
DATA = load_and_process_data()
print("Data loaded successfully.")

# Simple cache dictionary
request_cache = {}

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/college-finder.html')
def serve_college_finder():
    return app.send_static_file('college-finder.html')

@app.route('/api/colleges', methods=['POST'])
def get_colleges():
    if DATA is None or DATA.empty:
        return jsonify({"error": "Server data not loaded yet"}), 500
        
    req = request.json
    rank = req.get('rank')
    
    # Cast / category mapping
    category = req.get('category') or req.get('caste')
    if not category:
        return jsonify({"error": "category or caste is required"}), 400
        
    # Standardize old legacy front-end keys to database categories if needed
    category_map = {
        'OC': 'OC_BOYS',
        'OC-G': 'OC_GIRLS',
        'BC-A': 'BC_A_BOYS',
        'BC-B': 'BC_B_BOYS',
        'BC-C': 'BC_C_BOYS',
        'BC-D': 'BC_D_BOYS',
        'SC': 'SC_I_BOYS',
        'ST': 'ST_BOYS',
        'EWS': 'EWS_BOYS'
    }
    if category in category_map:
        category = category_map[category]
        
    branches = req.get('branches', [])
    
    # Cache key based on processed parameters
    cache_payload = {
        "rank": rank,
        "category": category,
        "branches": sorted(branches) if branches else []
    }
    cache_key = hashlib.md5(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
    if cache_key in request_cache:
        return jsonify(request_cache[cache_key])
        
    df = DATA[DATA['category'] == category]
    
    # Resolve mapped branches
    mapped_branches = []
    if branches:
        for b in branches:
            if b in BRANCH_MAPPING:
                mapped_branches.extend(BRANCH_MAPPING[b])
            else:
                mapped_branches.append(b)
        df = df[df['branch_code'].isin(mapped_branches)]
        
    results = []
    
    for _, row in df.iterrows():
        pred_cutoff = row['predicted_cutoff']
        if pd.isna(pred_cutoff):
            continue
            
        # If rank is provided, determine admission probability & bands
        if rank is not None:
            rank_val = int(rank)
            if rank_val <= pred_cutoff * 0.80:
                band = "SAFE"
                prob = 90
            elif rank_val <= pred_cutoff * 1.00:
                band = "MODERATE"
                prob = 65
            elif rank_val <= pred_cutoff * 1.10:
                band = "RISKY"
                prob = 30
            else:
                continue # Out of range
        else:
            band = "ALL"
            prob = 100
            
        latest_str_score = min(100, 100000 / pred_cutoff) if pred_cutoff > 0 else 0
        trend_score = 100 if row['trend_direction'] == 'RELAXING' else (50 if row['trend_direction'] == 'STABLE' else 0)
        branch_priority = 100 if row['branch_code'] in mapped_branches else 50
        college_reputation = 75
        
        score = (0.40 * prob) + (0.25 * latest_str_score) + (0.15 * trend_score) + (0.10 * branch_priority) + (0.10 * college_reputation)
        
        results.append({
            # Standard specification keys
            "college": row['institute_name'],
            "branch": row['branch_code'],
            "city": row['place'],
            "predicted_cutoff": int(pred_cutoff),
            "admission_probability": prob,
            "confidence_score": round(row['confidence_score'], 1),
            "trend": row['trend_direction'],
            "category": category,
            
            # Frontend compatibility keys
            "name": row['institute_name'],
            "type": row['college_type'],
            "branch_label": row['branch_name'],
            "cutoff": int(pred_cutoff),
            "can_get": bool(rank <= pred_cutoff) if rank is not None else True,
            "close": bool(pred_cutoff < rank <= pred_cutoff * 1.10) if rank is not None else False,
            
            # Extra analytical fields
            "college_code": row['institute_code'],
            "band": band,
            "score": score,
            "history": {
                "phase1": int(row['phase1_cutoff']) if not pd.isna(row['phase1_cutoff']) else None,
                "phase2": int(row['phase2_cutoff']) if not pd.isna(row['phase2_cutoff']) else None,
                "phase3": int(row['phase3_cutoff']) if not pd.isna(row['phase3_cutoff']) else None,
            }
        })
        
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top_safe = [r for r in results if r['band'] == 'SAFE']
    top_moderate = [r for r in results if r['band'] == 'MODERATE']
    dream = [r for r in results if r['band'] == 'RISKY']
    
    response_data = {
        "title": "🎯 Colleges Coming to You" if rank is not None else "🏛️ Colleges & Cutoffs",
        "subtitle": f"Based on your rank {rank} in {category}" if rank is not None else f"All options for category {category}",
        "colleges": results,
        "top_safe_colleges": top_safe[:10],
        "top_moderate_colleges": top_moderate[:10],
        "dream_colleges": dream[:10],
        "summary": {
            "total_matches": len(results),
            "top_safe_count": len(top_safe),
            "top_moderate_count": len(top_moderate),
            "dream_count": len(dream)
        }
    }
    
    request_cache[cache_key] = response_data
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
