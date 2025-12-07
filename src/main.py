#!/usr/bin/env python3
"""
AI Agent System - Main LangGraph Workflow

Orchestrates Manager and Generator agents for neural architecture generation.
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from agents.generator import generator_agent
from agents.manager import manager_agent


def create_workflow():
    """Create and compile the LangGraph workflow."""
    
    # Initialize workflow with our state
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("manager", manager_agent)
    workflow.add_node("generator", generator_agent)
    
    # Define workflow: Manager → Generator → End
    workflow.set_entry_point("manager")
    workflow.add_edge("manager", "generator")
    workflow.add_edge("generator", END)
    
    # Compile the workflow
    return workflow.compile()


def main():
    """Run the AI agent workflow."""
    
    print("=" * 60)
    print("🤖 AI AGENT SYSTEM - Neural Architecture Generation")
    print("=" * 60)
    
    # Create workflow
    app = create_workflow()
    
    # Initial state
    initial_state = {
        'epoch': 0,
        'conf_key': 'improve_classification_only',
        'base_model_name': 'resnet',
        'model_code': None,
        'accuracy': None,
        'loss': None,
        'metrics': None,
        'predicted_final_accuracy': None,
        'estimated_epochs': None,
        'gpu_available': True,
        'next_action': '',
        'found_in_cache': False,
        'experiment_id': 'exp_001',
        'status': 'pending'
    }
    
    print("\n📊 Starting workflow with initial state...")
    print(f"Epoch: {initial_state['epoch']}")
    print(f"Config: {initial_state['conf_key']}")
    
    # Run workflow
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ WORKFLOW COMPLETE!")
        print("=" * 60)
        print(f"Status: {result['status']}")
        print(f"Accuracy: {result.get('accuracy')}")
        print(f"Model code length: {len(result.get('model_code', ''))} chars")
        print(f"Found in cache: {result['found_in_cache']}")
        
    except Exception as e:
        print(f"\n❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()