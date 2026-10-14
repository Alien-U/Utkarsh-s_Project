import pandas as pd
import numpy as np
import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY" #For testing, you can use your own Groq API key. For security purposes, I am not pasting mine.

def process_and_evaluate_freight():
    print("Loading datasets...")
    df = pd.read_csv('shipment_records.csv')
    notes_df = pd.read_csv('context_notes.csv')
    notes_df['date'] = pd.to_datetime(notes_df['date'])

    print("Calculating baselines...")
    df['route'] = df['origin'] + '-' + df['destination']
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['week_of'] = df['shipment_date'] - pd.to_timedelta(df['shipment_date'].dt.dayofweek, unit='d')
    
    df['cost_per_tonne_km'] = df['freight_cost_inr'] / (df['quantity_tonnes'] * df['distance_km'])
    
    weekly = df.groupby(['route', 'route_type', 'week_of'])['cost_per_tonne_km'].mean().reset_index()
    weekly = weekly.sort_values(by=['route', 'week_of'])
    
    weekly['historical_avg'] = weekly.groupby('route')['cost_per_tonne_km'].transform(
        lambda x: x.shift(1).rolling(window=8, min_periods=1).mean()
    )
    
    peers = weekly.groupby(['route_type', 'week_of']).agg(
        total_type_cost=('cost_per_tonne_km', 'sum'),
        route_count=('route', 'count')
    ).reset_index()
    weekly = weekly.merge(peers, on=['route_type', 'week_of'], how='left')
    
    # Mathematical logic
    weekly['peer_avg'] = np.where(
        weekly['route_count'] > 1,
        (weekly['total_type_cost'] - weekly['cost_per_tonne_km']) / (weekly['route_count'] - 1),
        np.nan
    )
    
    # Evaluation Loop
    print("Evaluating flagged routes using AI...")
    THRESHOLD = 0.10
    output_rows = []
    
    # Initialize gpt model via Groq
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    
    prompt_template = PromptTemplate(
        input_variables=["route", "date", "notes_context"],
        template="""
        You are a logistics assistant. Evaluate if the cost increase for route {route} around the date {date} is justified by the provided notes.
        Notes context: {notes_context}
        
        Rules:
        1. If a note perfectly explains a cost increase (e.g., fuel hike, surcharge, disruption) for this route and time, it is justified.
        2. If no notes explain an increase, or notes say costs were NOT affected, it is NOT justified.
        
        Respond ONLY with a valid JSON object matching this schema:
        {{"flagged": "Yes" or "No (justified)", "matched_note_id": "Note ID (e.g. N002) or blank if unexplained", "reason": "Short plain English explanation"}}
        """
    )
    chain = prompt_template | llm

    total_llm_calls = 0

    for _, row in weekly.iterrows():
        hist_chg = ((row['cost_per_tonne_km'] - row['historical_avg']) / row['historical_avg']) * 100
        peer_chg = ((row['cost_per_tonne_km'] - row['peer_avg']) / row['peer_avg']) * 100
        
        vs_hist_str = f"{hist_chg:+.1f}% vs this route's past average" if pd.notna(hist_chg) else "No history"
        vs_peer_str = f"{peer_chg:+.1f}% vs similar-length routes this week" if pd.notna(peer_chg) else "No peers"
        
        if pd.notna(hist_chg) and pd.notna(peer_chg) and hist_chg > (THRESHOLD * 100) and peer_chg > (THRESHOLD * 100):
            relevant_notes = notes_df[
                (notes_df['applies_to'].isin([row['route'], 'All Routes'])) & 
                (notes_df['date'] <= row['week_of'])
            ]
            context_str = relevant_notes.to_string(index=False) if not relevant_notes.empty else "No relevant notes found."
            
            response = chain.invoke({
                "route": row['route'], 
                "date": row['week_of'].strftime('%Y-%m-%d'), 
                "notes_context": context_str
            })
            total_llm_calls += 1
            
            try:
                ai_data = json.loads(response.content)
                flagged = ai_data.get('flagged', 'Yes')
                note_id = ai_data.get('matched_note_id', '')
                reason = ai_data.get('reason', 'Unexplained rise.')
            except json.JSONDecodeError:
                flagged = "Yes"
                note_id = ""
                reason = "Cost rise looks unexplained and worth a human review."
        else:
            flagged = "No"
            note_id = ""
            reason = "Cost is stable; in line with historical and peer baselines."
            
        output_rows.append({
            'route': row['route'],
            'week_of': row['week_of'].strftime('%Y-%m-%d'),
            'cost_per_tonne_km': round(row['cost_per_tonne_km'], 2),
            'vs_own_history': vs_hist_str,
            'vs_similar_routes': vs_peer_str,
            'flagged': flagged,
            'matched_note_id': note_id,
            'reason': reason
        })

    # Output Generation
    final_df = pd.DataFrame(output_rows)
    final_df.to_csv('sample_output_format.csv', index=False)
    print(f"Run complete. {total_llm_calls} LLM calls made. Output saved to sample_output_format.csv.")

if __name__ == "__main__":
    process_and_evaluate_freight()