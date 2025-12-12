#!/usr/bin/env python3
"""
Test Follow-up Questions in Detailed Inquiry Stage
"""

import asyncio
import sys
import os

sys.path.append('/mnt/project')

async def test_followup_questions():
    print("🧪 Testing FOLLOW-UP QUESTIONS in Detailed Inquiry...")
    print("=" * 60)
    
    try:
        from conversation_engine import start_conversation, process_message
        
        conv_id = "test_followup_001"
        
        # Quick setup to detailed inquiry stage
        print("📋 SETUP: Getting to detailed inquiry stage...")
        await start_conversation(conv_id)
        await process_message(conv_id, "నా పేరు రాము")
        await process_message(conv_id, "మా గ్రామం రామారావుపేట")
        await process_message(conv_id, "వైద్య సేవలలో సమస్య ఉంది")
        print("✅ Setup complete - should be in detailed_inquiry stage")
        print()
        
        # Test follow-up questions
        print("🔍 TESTING FOLLOW-UP QUESTIONS:")
        print()
        
        # First detailed answer
        print("1️⃣ FIRST DETAILED ANSWER:")
        response1 = await process_message(conv_id, "మందులు సమయానికి రాకపోవడం")
        stage1 = response1['ai_response']['current_stage']
        ai_text1 = response1['ai_response']['telugu_response']
        
        print(f"📢 AI Response: {ai_text1}")
        print(f"📊 Stage: {stage1}")
        print(f"✅ Still in detailed_inquiry: {'YES' if stage1 == 'detailed_inquiry' else 'NO'}")
        print(f"✅ Asks follow-up question: {'YES' if any(word in ai_text1 for word in ['వైద్య', 'ఆస్పత్రి', 'డాక్టర్']) else 'NO'}")
        print()
        
        # Second detailed answer (if still in detailed inquiry)
        if stage1 == 'detailed_inquiry':
            print("2️⃣ SECOND DETAILED ANSWER:")
            response2 = await process_message(conv_id, "డాక్టర్ రాకపోవడం కూడా సమస్య")
            stage2 = response2['ai_response']['current_stage']
            ai_text2 = response2['ai_response']['telugu_response']
            
            print(f"📢 AI Response: {ai_text2}")
            print(f"📊 Stage: {stage2}")
            print(f"✅ Now in confirmation: {'YES' if stage2 == 'confirmation' else 'NO'}")
            print()
        else:
            print("❌ Jumped to confirmation too early - no second follow-up question!")
            print()
        
        print("🎯 EXPECTED BEHAVIOR:")
        print("1. First answer → AI asks second medical question (stays in detailed_inquiry)")
        print("2. Second answer → AI asks confirmation (progresses to confirmation)")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_followup_questions())