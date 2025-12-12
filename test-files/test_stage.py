#!/usr/bin/env python3
"""
Quick Test for Stage Progression Fix
"""

import asyncio
import sys
import os

# Add project directory to path
sys.path.append('/mnt/project')

async def test_stage_progression():
    print("🧪 Testing STAGE PROGRESSION FIX...")
    print("=" * 50)
    
    try:
        from conversation_engine import start_conversation, process_message
        
        conv_id = "test_stage_fix_001"
        
        # Start conversation
        print("1️⃣ START: Should be in GREETING stage")
        start_result = await start_conversation(conv_id)
        initial_response = start_result['initial_response']['telugu_response']
        print(f"AI: {initial_response}")
        expected_greeting = "నమస్కారం! నేను జన స్పందన AI. మీ పేరు ఏమిటి?"
        print(f"✅ Correct greeting: {'YES' if expected_greeting in initial_response else 'NO'}")
        print()
        
        # Give name - should progress to NAME_COLLECTION and ask for village
        print("2️⃣ NAME: Should progress to NAME_COLLECTION and ask for village")
        name_response = await process_message(conv_id, "నా పేరు రాము")
        ai_text = name_response['ai_response']['telugu_response']
        stage = name_response['ai_response']['current_stage']
        print(f"AI: {ai_text}")
        print(f"Stage: {stage}")
        expected_village_q = "స్వాగతం రాము అన్న! మీరు ఏ గ్రామం నుండి వచ్చారు?"
        print(f"✅ Village question: {'YES' if expected_village_q in ai_text else 'NO'}")
        print(f"✅ Correct stage: {'YES' if stage == 'name_collection' else 'NO'}")
        print()
        
        # Give village - should progress to SECTOR_IDENTIFICATION and ask for sectors
        print("3️⃣ VILLAGE: Should progress to SECTOR_IDENTIFICATION and list sectors")
        village_response = await process_message(conv_id, "మా గ్రామం రామారావుపేట")
        ai_text = village_response['ai_response']['telugu_response']
        stage = village_response['ai_response']['current_stage']
        print(f"AI: {ai_text}")
        print(f"Stage: {stage}")
        has_sectors = all(sector in ai_text for sector in ['వైద్య సేవలు', 'మౌలిక వసతులు', 'విద్యా సేవలు', 'ప్రభుత్వ పథకాలు'])
        print(f"✅ Has all 4 sectors: {'YES' if has_sectors else 'NO'}")
        print(f"✅ Correct stage: {'YES' if stage == 'sector_identification' else 'NO'}")
        print()
        
        # Choose sector - should progress to DETAILED_INQUIRY and ask specific question
        print("4️⃣ SECTOR: Should progress to DETAILED_INQUIRY and ask medical question")
        sector_response = await process_message(conv_id, "వైద్య సేవలలో సమస్య ఉంది")
        ai_text = sector_response['ai_response']['telugu_response']
        stage = sector_response['ai_response']['current_stage']
        print(f"AI: {ai_text}")
        print(f"Stage: {stage}")
        has_medical_question = any(word in ai_text for word in ['ఆస్పత్రి', 'వైద్యుడు', 'మందులు', 'వైద్య సేవలు'])
        print(f"✅ Medical question: {'YES' if has_medical_question else 'NO'}")
        print(f"✅ Correct stage: {'YES' if stage == 'detailed_inquiry' else 'NO'}")
        print()
        
        print("🎯 SUMMARY:")
        print("If all checks show 'YES', the stage progression is now working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stage_progression())