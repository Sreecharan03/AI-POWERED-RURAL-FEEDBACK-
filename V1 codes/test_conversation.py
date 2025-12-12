#!/usr/bin/env python3
"""
Test Script for Fixed JanSpandana.AI Conversation Engine
Run this to verify the conversation flow works correctly

IMPORTANT: Replace your conversation_engine.py with the fixed version first!
"""

import asyncio
import sys
import os

# Add project directory to path
sys.path.append('/mnt/project')

async def test_fixed_conversation():
    print("🚀 JanSpandana.AI Conversation Engine Test")
    print("=" * 60)
    print()
    
    print("📋 INSTRUCTIONS:")
    print("1. Copy conversation_engine_fixed.py to your project folder")
    print("2. Rename your current conversation_engine.py to conversation_engine_backup.py") 
    print("3. Rename conversation_engine_fixed.py to conversation_engine.py")
    print("4. Run this test")
    print()
    
    try:
        # Import the conversation engine (should be the fixed version)
        from conversation_engine import start_conversation, process_message, end_conversation_session
        
        print("✅ Successfully imported conversation engine")
        print()
        
        # Test conversation flow
        conv_id = "test_fixed_conversation_001"
        
        print("📋 Test Scenario:")
        print("1. Start conversation → Should ask for name")
        print("2. Give name 'నా పేరు రాము' → Should ask for village")  
        print("3. Give village 'మా గ్రామం రామారావుపేట' → Should ask for sector")
        print("4. Choose sector 'వైద్య సేవలు' → Should ask detailed question")
        print()
        
        # 1. Start conversation
        print("🟡 Step 1: Starting conversation...")
        start_result = await start_conversation(conv_id)
        
        if start_result['session_started']:
            initial_response = start_result['initial_response']['telugu_response']
            print(f"✅ Start successful")
            print(f"📢 AI: {initial_response}")
            expected = "నమస్కారం! నేను జన స్పందన AI. మీ పేరు ఏమిటి?"
            print(f"🎯 Expected: '{expected}'")
            print(f"✅ Match: {'YES' if expected in initial_response else 'NO'}")
            print()
        else:
            print("❌ Failed to start conversation")
            return
        
        # 2. Give name
        print("🟡 Step 2: Giving name...")
        name_response = await process_message(conv_id, "నా పేరు రాము")
        
        if name_response['success']:
            ai_text = name_response['ai_response']['telugu_response']
            stage = name_response['ai_response']['current_stage']
            
            print(f"📢 AI: {ai_text}")
            expected = "స్వాగతం రాము అన్న! మీరు ఏ గ్రామం నుండి వచ్చారు?"
            print(f"🎯 Expected: '{expected}'")
            print(f"📊 Stage: {stage}")
            print(f"✅ Village Question: {'YES' if expected in ai_text else 'NO'}")
            print(f"✅ Correct Stage: {'YES' if stage == 'name_collection' else 'NO'}")
            print()
        else:
            print(f"❌ Name processing failed: {name_response}")
            return
        
        # 3. Give village
        print("🟡 Step 3: Giving village...")
        village_response = await process_message(conv_id, "మా గ్రామం రామారావుపేట")
        
        if village_response['success']:
            ai_text = village_response['ai_response']['telugu_response']
            stage = village_response['ai_response']['current_stage']
            
            print(f"📢 AI: {ai_text}")
            print(f"🎯 Expected: Lists 4 sectors (వైద్య సేవలు, మౌలిక వసతులు, విద్యా సేవలు, ప్రభుత్వ పథకాలు)")
            print(f"📊 Stage: {stage}")  
            has_all_sectors = all(sector in ai_text for sector in ['వైద్య సేవలు', 'మౌలిక వసతులు', 'విద్యా సేవలు', 'ప్రభుత్వ పథకాలు'])
            print(f"✅ Sector Options: {'YES' if has_all_sectors else 'NO'}")
            print(f"✅ Correct Stage: {'YES' if stage == 'sector_identification' else 'NO'}")
            print()
        else:
            print(f"❌ Village processing failed: {village_response}")
            return
            
        # 4. Choose sector
        print("🟡 Step 4: Choosing sector...")
        sector_response = await process_message(conv_id, "వైద్య సేవలలో సమస్య ఉంది")
        
        if sector_response['success']:
            ai_text = sector_response['ai_response']['telugu_response']
            stage = sector_response['ai_response']['current_stage']
            
            print(f"📢 AI: {ai_text}")
            print(f"🎯 Expected: Ask specific medical services question")
            print(f"📊 Stage: {stage}")
            has_medical_question = any(word in ai_text for word in ['ఆస్పత్రి', 'వైద్యుడు', 'మందులు', 'వైద్య సేవలు'])
            print(f"✅ Medical Question: {'YES' if has_medical_question else 'NO'}")
            print(f"✅ Correct Stage: {'YES' if stage == 'detailed_inquiry' else 'NO'}")
            print()
        else:
            print(f"❌ Sector processing failed: {sector_response}")
            return
            
        # 5. Answer detailed question
        print("🟡 Step 5: Answering detailed question...")
        detail_response = await process_message(conv_id, "మందులు సమయానికి రావడం లేదు")
        
        if detail_response['success']:
            ai_text = detail_response['ai_response']['telugu_response']
            stage = detail_response['ai_response']['current_stage']
            
            print(f"📢 AI: {ai_text}")
            print(f"📊 Stage: {stage}")
            print()
            
        # End conversation
        print("🟡 Step 6: Ending conversation...")
        end_result = await end_conversation_session(conv_id)
        
        if end_result['success']:
            print("✅ Conversation ended successfully")
            print()
            
            # Summary
            print("📊 CONVERSATION SUMMARY:")
            summary = end_result.get('summary', {})
            print(f"👤 User Name: {summary.get('user_name', 'Not collected')}")
            print(f"🏘️ Village: {summary.get('village_name', 'Not collected')}")
            print(f"🎯 Sector: {summary.get('identified_sector', 'Not identified')}")
            print(f"💬 Total Messages: {summary.get('total_questions', 0)}")
            print(f"📝 Issues Collected: {len(summary.get('issues_collected', []))}")
        else:
            print(f"❌ End conversation failed: {end_result}")
        
        print()
        print("🎉 TEST COMPLETED!")
        print("✅ If all 'Correct Stage' checks show 'YES', the conversation flow is working!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure to replace your conversation_engine.py with conversation_engine_fixed.py")
        print("   1. Backup: mv conversation_engine.py conversation_engine_backup.py")
        print("   2. Copy: cp conversation_engine_fixed.py conversation_engine.py")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 JanSpandana.AI Conversation Engine Test")
    print()
    
    # Run the test
    asyncio.run(test_fixed_conversation())