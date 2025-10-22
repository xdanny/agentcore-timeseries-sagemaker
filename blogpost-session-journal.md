# Building a Session Journal Tool with S3: Tracing the Chaos in Your AI Agent

So you're building an AI agent system with AWS Bedrock AgentCore and Strands, and everything's going great... until it's not. Your agent is calling tools left and right, spinning up Code Interpreter sessions, hitting SageMaker endpoints, and somewhere in that beautiful chaos, something breaks.

"What just happened?" you ask. "Which tool did what? When did that session start?"

Yeah, I've been there. Let me show you how I built a session journal tool that traces everything, uses S3 as a backing store, and actually keeps sandboxed tools synchronized. Spoiler: it's not as hard as it sounds.

## The Problem: Visibility in the Void

Here's the thing about agent systems - they're async, distributed, and full of side effects. You've got:

- **Code Interpreter sessions** that spin up and die
- **SageMaker training jobs** that take forever
- **S3 operations** happening all over the place
- **Tool calls** that chain into each other

Without proper tracing, debugging feels like archaeology. You're digging through CloudWatch logs, piecing together timestamps, trying to figure out which session did what.

We needed something better. Something that could:

1. **Track every tool invocation** - what was called, when, with what params
2. **Trace session lifecycles** - start, actions, cleanup
3. **Persist to S3** - because everything else is there anyway
4. **Work in sandboxed environments** - Code Interpreter can't use boto3 directly
5. **Synchronize across tools** - so different tools can read each other's journals

## The Solution: Session Journal Tool

The approach is actually pretty elegant. We create a Strands tool that acts like a distributed append-only log. Every tool can write to it, and it all syncs to S3.

### Architecture Overview

```
┌─────────────────┐
│   Agent Core    │
│  (has boto3)    │
└────────┬────────┘
         │
         ├─► Tool A ──┐
         │            │
         ├─► Tool B ──┼─► Session Journal ─► S3
         │            │   (orchestrates)
         ├─► Tool C ──┘
         │
         └─► Code Interpreter
             (sandboxed, uses markers)
```

The journal tool sits between your tools and S3, handling the complexity of:
- Appending to existing logs (S3 is immutable, so we download → append → upload)
- Synchronizing across concurrent tool calls (using S3 versioning)
- Working around sandbox limitations (marker string pattern)

### Step 1: Define the Journal Tool

First, let's create the actual tool. I'm keeping it simple but powerful:

```python
# agents/session_journal.py
import json
import boto3
import os
from datetime import datetime
from typing import Optional, Dict, Any
from strands import tool

BUCKET = os.environ.get('JOURNAL_BUCKET',
    f"session-journal-{os.environ.get('AWS_ACCOUNT_ID')}")
S3_CLIENT = None

def get_s3_client():
    """Lazy initialization of S3 client"""
    global S3_CLIENT
    if S3_CLIENT is None:
        S3_CLIENT = boto3.client('s3')
    return S3_CLIENT

@tool
def write_journal_entry(
    session_id: str,
    event_type: str,
    tool_name: str,
    details: Optional[Dict[str, Any]] = None,
    level: str = "INFO"
) -> str:
    """
    Write an entry to the session journal stored in S3.

    This tool traces all tool invocations, session events, and system calls.
    Each session gets its own journal file in S3 that grows over time.

    Args:
        session_id: Unique identifier for the session (e.g., CI session ID)
        event_type: Type of event (TOOL_START, TOOL_END, SESSION_START, etc.)
        tool_name: Name of the tool being traced
        details: Optional dictionary with additional context
        level: Log level (INFO, WARNING, ERROR, DEBUG)

    Returns:
        JSON string with journal entry confirmation and S3 path
    """
    s3 = get_s3_client()

    # Create journal entry
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "event_type": event_type,
        "tool_name": tool_name,
        "level": level,
        "details": details or {}
    }

    # S3 key for this session's journal
    key = f"journals/{session_id}/journal.jsonl"

    try:
        # Try to download existing journal
        response = s3.get_object(Bucket=BUCKET, Key=key)
        existing_content = response['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        # First entry for this session
        existing_content = ""

    # Append new entry (JSONL format - one JSON per line)
    new_content = existing_content + json.dumps(entry) + "\n"

    # Upload back to S3 with versioning
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=new_content.encode('utf-8'),
        ContentType='application/x-jsonlines'
    )

    return json.dumps({
        "status": "success",
        "entry": entry,
        "s3_path": f"s3://{BUCKET}/{key}",
        "entry_count": len(new_content.strip().split('\n'))
    })

@tool
def read_journal(session_id: str, filter_event_type: Optional[str] = None) -> str:
    """
    Read journal entries for a session.

    Args:
        session_id: Session to read journal for
        filter_event_type: Optional filter (e.g., only "TOOL_START" events)

    Returns:
        JSON string with list of journal entries
    """
    s3 = get_s3_client()
    key = f"journals/{session_id}/journal.jsonl"

    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        content = response['Body'].read().decode('utf-8')

        # Parse JSONL
        entries = [json.loads(line) for line in content.strip().split('\n') if line]

        # Optional filtering
        if filter_event_type:
            entries = [e for e in entries if e['event_type'] == filter_event_type]

        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "entry_count": len(entries),
            "entries": entries
        })
    except s3.exceptions.NoSuchKey:
        return json.dumps({
            "status": "not_found",
            "session_id": session_id,
            "message": "No journal found for this session"
        })
```

That's the core. Pretty straightforward, right? A few things to notice:

- **JSONL format** - One JSON object per line. Super easy to append without parsing everything.
- **Versioning-ready** - S3 versioning handles concurrent writes if you enable it.
- **Lazy S3 client** - We don't initialize boto3 until needed.

### Step 2: Instrument Your Existing Tools

Now the fun part - adding journal entries to your tools. Here's how I retrofitted our EDA tool:

```python
# agents/advanced_eda_agent.py
from strands import tool
from .session_journal import write_journal_entry
from .code_interpreter_utils import invoke_with_session
import uuid

@tool
def run_advanced_eda(
    dataset_s3_path: str,
    time_column: str = None,
    value_column: str = None
) -> str:
    """Run advanced exploratory data analysis on time series data."""

    # Generate a unique session ID for tracing
    session_id = f"eda_{uuid.uuid4().hex[:8]}"

    # Log tool start
    write_journal_entry(
        session_id=session_id,
        event_type="TOOL_START",
        tool_name="run_advanced_eda",
        details={
            "dataset_s3_path": dataset_s3_path,
            "time_column": time_column,
            "value_column": value_column
        }
    )

    try:
        # Your actual tool logic here
        code = f'''
import pandas as pd
import subprocess
import sys

# Log that we're starting data download (from inside sandbox)
print("===JOURNAL_ENTRY===")
print('{{"event_type": "DATA_DOWNLOAD_START", "path": "{dataset_s3_path}"}}')
print("===JOURNAL_END===")

# Download data
subprocess.run(['aws', 's3', 'cp', '{dataset_s3_path}', '/tmp/data.csv'],
               check=True, capture_output=True)

df = pd.read_csv('/tmp/data.csv')

print("===JOURNAL_ENTRY===")
print('{{"event_type": "DATA_LOADED", "rows": {len(df)}, "cols": {len(df.columns)}}}')
print("===JOURNAL_END===")

# Run analysis...
# (your existing EDA code)

print("===JOURNAL_ENTRY===")
print('{{"event_type": "ANALYSIS_COMPLETE"}}')
print("===JOURNAL_END===")
'''

        # Execute in Code Interpreter
        response = invoke_with_session(code, language='python')
        output = response['output']

        # Extract journal entries from sandboxed environment
        entries = extract_journal_entries(output)
        for entry_data in entries:
            write_journal_entry(
                session_id=session_id,
                event_type=entry_data.get('event_type', 'CODE_INTERPRETER_EVENT'),
                tool_name="run_advanced_eda",
                details=entry_data
            )

        # Log success
        write_journal_entry(
            session_id=session_id,
            event_type="TOOL_END",
            tool_name="run_advanced_eda",
            details={"status": "success"}
        )

        return output

    except Exception as e:
        # Log failure
        write_journal_entry(
            session_id=session_id,
            event_type="TOOL_ERROR",
            tool_name="run_advanced_eda",
            details={"error": str(e)},
            level="ERROR"
        )
        raise

def extract_journal_entries(output: str) -> list:
    """Extract journal entries from Code Interpreter output using markers."""
    entries = []
    lines = output.split('\n')

    i = 0
    while i < len(lines):
        if '===JOURNAL_ENTRY===' in lines[i]:
            # Find the end marker
            j = i + 1
            while j < len(lines) and '===JOURNAL_END===' not in lines[j]:
                j += 1

            # Extract the JSON between markers
            if j < len(lines):
                entry_json = '\n'.join(lines[i+1:j])
                try:
                    entries.append(json.loads(entry_json))
                except json.JSONDecodeError:
                    pass  # Skip malformed entries
            i = j + 1
        else:
            i += 1

    return entries
```

### Step 3: The Marker Pattern for Sandboxed Environments

This is the clever bit. Code Interpreter doesn't have boto3 access, but it *can* print to stdout. So we use a marker pattern:

```python
# Inside Code Interpreter (sandboxed):
print("===JOURNAL_ENTRY===")
print(json.dumps({"event_type": "SOMETHING_HAPPENED", "details": "..."}))
print("===JOURNAL_END===")

# Outside (in AgentCore):
output = code_interpreter.invoke(code)
entries = extract_journal_entries(output)
for entry in entries:
    write_journal_entry(session_id, **entry)
```

This pattern lets sandboxed code communicate structured events back to the orchestration layer, where we have S3 access. It's the same pattern we use for HTML report extraction.

### Step 4: Synchronization Between Tools

Now here's where it gets interesting. Say you have a pipeline:

1. **Tool A** starts a SageMaker training job
2. **Tool B** (later) needs to check if training completed
3. **Tool C** generates a report using the trained model

They all need to share state. The journal becomes our source of truth:

```python
# agents/sagemaker_orchestrator.py
from .session_journal import write_journal_entry, read_journal
import json

@tool
def start_training_workflow(dataset_path: str, session_id: str) -> str:
    """Start a training workflow and log it."""

    # Log workflow start
    write_journal_entry(
        session_id=session_id,
        event_type="WORKFLOW_START",
        tool_name="start_training_workflow",
        details={"dataset_path": dataset_path}
    )

    # Start training job
    job_name = create_sagemaker_training_job(dataset_path)

    # Log the job name so other tools can find it
    write_journal_entry(
        session_id=session_id,
        event_type="TRAINING_JOB_STARTED",
        tool_name="start_training_workflow",
        details={"job_name": job_name, "status": "InProgress"}
    )

    return json.dumps({"job_name": job_name, "session_id": session_id})

@tool
def check_training_status(session_id: str) -> str:
    """Check status of training job from journal."""

    # Read the journal to find the training job
    journal_response = read_journal(session_id, filter_event_type="TRAINING_JOB_STARTED")
    journal_data = json.loads(journal_response)

    if not journal_data.get('entries'):
        return json.dumps({"error": "No training job found in session"})

    # Get the most recent training job
    last_entry = journal_data['entries'][-1]
    job_name = last_entry['details']['job_name']

    # Check actual status in SageMaker
    sm_client = boto3.client('sagemaker')
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    status = response['TrainingJobStatus']

    # Log the status check
    write_journal_entry(
        session_id=session_id,
        event_type="TRAINING_STATUS_CHECK",
        tool_name="check_training_status",
        details={"job_name": job_name, "status": status}
    )

    return json.dumps({"job_name": job_name, "status": status})

@tool
def generate_report_from_trained_model(session_id: str) -> str:
    """Generate report using model from completed training."""

    # Read journal to find completed training
    journal_response = read_journal(session_id)
    journal_data = json.loads(journal_response)

    # Find the training job that completed
    training_events = [
        e for e in journal_data['entries']
        if e['event_type'] in ['TRAINING_JOB_STARTED', 'TRAINING_STATUS_CHECK']
    ]

    if not training_events:
        return json.dumps({"error": "No training history found"})

    # Get the job name
    job_name = training_events[-1]['details']['job_name']

    # Generate report...
    # (your report generation logic)

    write_journal_entry(
        session_id=session_id,
        event_type="REPORT_GENERATED",
        tool_name="generate_report_from_trained_model",
        details={"job_name": job_name, "report_path": report_s3_path}
    )

    return json.dumps({"report_path": report_s3_path})
```

See what's happening? The journal is our shared state store. Tools write events, and other tools read those events to coordinate. It's like a lightweight event sourcing system.

### Step 5: Handling Concurrency with S3 Versioning

S3 is eventually consistent, but we can handle concurrent writes with versioning:

```python
# Enable versioning on your journal bucket
def setup_journal_bucket():
    s3 = boto3.client('s3')
    bucket_name = BUCKET

    # Create bucket if it doesn't exist
    try:
        s3.create_bucket(Bucket=bucket_name)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Enable versioning for conflict detection
    s3.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={'Status': 'Enabled'}
    )

    # Optional: Set lifecycle policy to archive old versions
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration={
            'Rules': [{
                'Id': 'ArchiveOldVersions',
                'Status': 'Enabled',
                'NoncurrentVersionTransitions': [{
                    'NoncurrentDays': 30,
                    'StorageClass': 'GLACIER'
                }]
            }]
        }
    )
```

For true synchronization with optimistic locking:

```python
def write_journal_entry_with_retry(session_id, event_type, tool_name, details=None, level="INFO", max_retries=3):
    """Write with retry logic for concurrent modification."""
    s3 = get_s3_client()
    key = f"journals/{session_id}/journal.jsonl"

    for attempt in range(max_retries):
        try:
            # Get current version
            try:
                response = s3.get_object(Bucket=BUCKET, Key=key)
                existing_content = response['Body'].read().decode('utf-8')
                current_version_id = response['VersionId']
            except s3.exceptions.NoSuchKey:
                existing_content = ""
                current_version_id = None

            # Append new entry
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "session_id": session_id,
                "event_type": event_type,
                "tool_name": tool_name,
                "level": level,
                "details": details or {}
            }
            new_content = existing_content + json.dumps(entry) + "\n"

            # Upload with conditional write (if version matches)
            put_params = {
                'Bucket': BUCKET,
                'Key': key,
                'Body': new_content.encode('utf-8'),
                'ContentType': 'application/x-jsonlines'
            }

            # Only works if object hasn't changed
            if current_version_id:
                # Check if the current version is still the latest
                # This is a simplified version - real implementation would use ETags
                pass

            s3.put_object(**put_params)

            return json.dumps({
                "status": "success",
                "entry": entry,
                "s3_path": f"s3://{BUCKET}/{key}",
                "attempt": attempt + 1
            })

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff
            time.sleep(0.1 * (2 ** attempt))

    raise Exception("Failed to write journal entry after retries")
```

## Real-World Usage Example

Here's how it all comes together in a real workflow:

```python
# agent.py - Main orchestration
from strands import Agent
from agents.session_journal import write_journal_entry, read_journal
from agents.advanced_eda_agent import run_advanced_eda
from agents.sagemaker_orchestrator import start_training_workflow, check_training_status

agent = Agent(
    name="IntelligentForecastingAgent",
    tools=[
        write_journal_entry,
        read_journal,
        run_advanced_eda,
        start_training_workflow,
        check_training_status,
        # ... other tools
    ]
)

# User prompt: "Analyze this dataset and train a model"
session_id = f"workflow_{uuid.uuid4().hex[:12]}"

# Step 1: EDA (logs internally)
eda_result = run_advanced_eda(
    dataset_s3_path="s3://my-bucket/data.csv",
    time_column="date",
    value_column="sales"
)

# Step 2: Start training (logs job start)
training_result = start_training_workflow(
    dataset_path="s3://my-bucket/data.csv",
    session_id=session_id
)

# Step 3: Later, check status (reads from journal)
status_result = check_training_status(session_id=session_id)

# Step 4: View the full audit trail
journal = read_journal(session_id=session_id)
print(json.dumps(json.loads(journal), indent=2))
```

The journal output looks like:

```json
{
  "status": "success",
  "session_id": "workflow_a3b2c1d4e5f6",
  "entry_count": 8,
  "entries": [
    {
      "timestamp": "2025-10-22T14:32:01Z",
      "session_id": "workflow_a3b2c1d4e5f6",
      "event_type": "TOOL_START",
      "tool_name": "run_advanced_eda",
      "level": "INFO",
      "details": {
        "dataset_s3_path": "s3://my-bucket/data.csv",
        "time_column": "date",
        "value_column": "sales"
      }
    },
    {
      "timestamp": "2025-10-22T14:32:05Z",
      "event_type": "DATA_DOWNLOAD_START",
      "tool_name": "run_advanced_eda",
      "details": {"path": "s3://my-bucket/data.csv"}
    },
    {
      "timestamp": "2025-10-22T14:32:12Z",
      "event_type": "DATA_LOADED",
      "tool_name": "run_advanced_eda",
      "details": {"rows": 1000, "cols": 3}
    },
    {
      "timestamp": "2025-10-22T14:32:45Z",
      "event_type": "TOOL_END",
      "tool_name": "run_advanced_eda",
      "details": {"status": "success"}
    },
    {
      "timestamp": "2025-10-22T14:33:01Z",
      "event_type": "WORKFLOW_START",
      "tool_name": "start_training_workflow",
      "details": {"dataset_path": "s3://my-bucket/data.csv"}
    },
    {
      "timestamp": "2025-10-22T14:33:15Z",
      "event_type": "TRAINING_JOB_STARTED",
      "tool_name": "start_training_workflow",
      "details": {
        "job_name": "arima-job-2025-10-22-14-33-15",
        "status": "InProgress"
      }
    },
    {
      "timestamp": "2025-10-22T14:45:20Z",
      "event_type": "TRAINING_STATUS_CHECK",
      "tool_name": "check_training_status",
      "details": {
        "job_name": "arima-job-2025-10-22-14-33-15",
        "status": "Completed"
      }
    }
  ]
}
```

Beautiful, right? Complete traceability.

## Bonus: Visualization Dashboard

Want to see your journals visually? Here's a quick Streamlit app:

```python
# scripts/journal_viewer.py
import streamlit as st
import boto3
import json
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Session Journal Viewer", layout="wide")
st.title("Session Journal Viewer")

# List all sessions
s3 = boto3.client('s3')
bucket = "session-journal-us-east-1-123456789"

response = s3.list_objects_v2(Bucket=bucket, Prefix="journals/", Delimiter="/")
sessions = [p['Prefix'].split('/')[1] for p in response.get('CommonPrefixes', [])]

selected_session = st.selectbox("Select Session", sessions)

if selected_session:
    # Load journal
    key = f"journals/{selected_session}/journal.jsonl"
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj['Body'].read().decode('utf-8')

    entries = [json.loads(line) for line in content.strip().split('\n') if line]
    df = pd.DataFrame(entries)

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(df))
    col2.metric("Tools Used", df['tool_name'].nunique())
    col3.metric("Errors", len(df[df['level'] == 'ERROR']))
    col4.metric("Duration", f"{(df['timestamp'].max() - df['timestamp'].min()).seconds}s")

    # Timeline
    st.subheader("Event Timeline")

    # Color code by event type
    color_map = {
        'TOOL_START': 'green',
        'TOOL_END': 'blue',
        'TOOL_ERROR': 'red',
        'DATA_DOWNLOAD_START': 'orange',
        'TRAINING_JOB_STARTED': 'purple'
    }

    df['color'] = df['event_type'].map(color_map).fillna('gray')

    import plotly.express as px
    fig = px.scatter(df, x='timestamp', y='event_type', color='event_type',
                     hover_data=['tool_name', 'level'],
                     title="Event Timeline")
    st.plotly_chart(fig, use_container_width=True)

    # Details table
    st.subheader("Event Details")
    st.dataframe(df[['timestamp', 'event_type', 'tool_name', 'level']], use_container_width=True)

    # Expandable details
    for idx, row in df.iterrows():
        with st.expander(f"{row['timestamp']} - {row['event_type']}"):
            st.json(row['details'])
```

## Key Takeaways

1. **S3 as a database works** - For append-only logs, S3 is cheap, durable, and good enough. JSONL format makes it easy to append.

2. **Marker patterns bridge sandboxes** - When you can't use AWS SDKs inside Code Interpreter, use stdout with markers. Extract on the other side.

3. **Journals enable coordination** - Tools can read each other's journals to synchronize state without complex messaging systems.

4. **Instrumentation is easy** - Just add `write_journal_entry()` calls at key points. Wrap tools in try/catch to log errors.

5. **Version control prevents conflicts** - Enable S3 versioning and use retries for concurrent writes.

## What's Next?

Some ideas to extend this:

- **Query API**: Build a `query_journal(session_id, sql_query)` tool using DuckDB to query JSONL files
- **Real-time streaming**: Push journal entries to Kinesis for real-time dashboards
- **Alerting**: Lambda function that watches journals and sends alerts on errors
- **Cost tracking**: Log AWS service calls with estimated costs
- **Performance profiling**: Calculate time between TOOL_START and TOOL_END events

The session journal pattern has been a game-changer for our forecasting system. When something breaks at 2am, we can trace exactly what happened without digging through raw logs.

Give it a try in your Strands agent setup. Your future debugging self will thank you.

---

*Questions? Find me on GitHub or leave a comment below. Happy journaling!*
