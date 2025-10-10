#!/usr/bin/env python3
"""
Code Interpreter Session Management - Shared utility for all agents
Creates fresh session for each invocation and properly stops it when done
"""
import boto3

CODE_INTERPRETER_ID = 'forecasting_code_interpreter-2sBCjbrp6B'


def invoke_with_session(code, language='python'):
    """
    Invoke Code Interpreter with fresh session (start -> invoke -> stop)

    This pattern ensures:
    1. Always uses a fresh, active session
    2. Properly cleans up resources when done
    3. No stale session errors

    Args:
        code: Python code to execute
        language: Programming language (default: python)

    Returns:
        response: Code Interpreter response stream
    """
    client = boto3.client('bedrock-agentcore', region_name='us-east-1')
    session_id = None

    try:
        # Start fresh session
        start_response = client.start_code_interpreter_session(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionTimeoutSeconds=900  # 15 minutes
        )
        session_id = start_response['sessionId']

        # Invoke code
        response = client.invoke_code_interpreter(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=session_id,
            name='executeCode',
            arguments={
                'code': code,
                'language': language
            }
        )

        return response

    finally:
        # Always stop session when done (cleanup)
        if session_id:
            try:
                client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=CODE_INTERPRETER_ID,
                    sessionId=session_id
                )
            except Exception:
                pass  # Best effort cleanup


# Keep old function name for backwards compatibility
invoke_with_retry = invoke_with_session
